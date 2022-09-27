import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torchvision.models import resnet50
import torchvision.transforms as T
from einops import rearrange
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from PIL import Image
from pytorch_lightning import LightningModule
from utils.data_utils import denormalize, get_dataloader

from models.backbone_convnext import convnext_base
from models.backbone_vit import ViTExtractor
from models.counting_head import CountingHeadLinearProbe, CountingHeadRegression
from models.localisation_head import LocalsiationConvUpsample

from sklearn.decomposition import PCA


class CountingAnything(LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.seed = torch.tensor(CFG["seed"]).cuda()
        self.define_metrics(CFG)
        counting_resloution = 28
        self.bs = CFG["train_batch_size"]
        number_to_unfreeze = self.CFG["backbone_unfreeze_layers"]

        self.resize224 = T.Resize(224, T.InterpolationMode.NEAREST)
        self.resize224b = T.Resize(224, T.InterpolationMode.BILINEAR)

        if CFG["backbone"] == "vit_dino":
            # vit_small_patch8_224_dino-token_11_0_8
            feature_dim = 384
            vit_config = {
                "base_model": "vit_small_patch8_224_dino",
                "facet": "token",
                "layer": int(11),
                "bin": False,
                "stride": int(8),
            }
            self.backbone = ViTExtractor(vit_config)

            if number_to_unfreeze == -1:
                print("not freezing any of the counting backbone")
            else:
                model_dict = list(
                    {
                        int(k.split(".")[1])
                        for k, _ in self.backbone.named_parameters()
                        if k.startswith("blocks.")
                    }
                )
                if number_to_unfreeze != 0:
                    dont_freeze = model_dict[-number_to_unfreeze:]
                    dont_freeze = [f"blocks.{str(k)}." for k in dont_freeze]
                else:
                    dont_freeze = []

                dont_freeze_list = []
                for name, param in self.backbone.named_parameters():
                    in_dont_freeze = False
                    for df in dont_freeze:
                        if name.startswith(df):
                            in_dont_freeze = True
                    if not in_dont_freeze:
                        param.requires_grad = False
                    else:
                        dont_freeze_list.append(name)

                dont_freeze_list = set(
                    [v.split(".")[0] + v.split(".")[1] for v in dont_freeze_list]
                )
                print("not freezing from counting backbone", dont_freeze_list)

        elif CFG["backbone"] == "resnet":
            # resnet-50
            feature_dim = 512
            backbone = resnet50(pretrained=True)
            layers = list(backbone.children())[:-4]
            self.backbone = nn.Sequential(*layers)

            if number_to_unfreeze == -1:
                print("not freezing any of the counting backbone")
            else:
                print("freezing the counting backbone")
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False

        elif CFG["backbone"] == "convnext":  #
            # convnext_base
            feature_dim = 256
            self.backbone = convnext_base(pretrained=True)

            if number_to_unfreeze == -1:
                print("not freezing any of the counting backbone")
            else:
                print("freezing the counting backbone")
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False

        if self.CFG["use_counting_head"]:
            count_strategy = self.CFG["counting_head"].split("_")
            counting_head_type = count_strategy[0]

            if counting_head_type == "regress":
                head_complexity = count_strategy[2]
                self.ch = int(count_strategy[1][1:])  # starts with a "c"
                self.counting_head = CountingHeadRegression(
                    feature_dim,
                    counting_resloution,
                    c=self.ch,
                    complexity=head_complexity,
                )
            elif counting_head_type == "linear":
                dim_pred = 1
                self.counting_head = CountingHeadLinearProbe(
                    feature_dim, counting_resloution, dim_pred,
                )

        if self.CFG["use_localisation_head"]:
            localisation_resloution = 224
            localisation_strategy = self.CFG["localisation_head"].split("_")
            localisation_head_type = localisation_strategy[0]

            if localisation_head_type == "conv":
                self.localisation_head = LocalsiationConvUpsample(feature_dim)
            if self.CFG["localisation_loss"] == "MSE":
                self.loc_loss = nn.MSELoss()

        if (
            self.CFG["resume_path"] != ""
            or self.CFG["resume_counting_head_path"] != ""
            or self.CFG["resume_path_localisation"] != ""
        ):
            self.load_pretrained_model()
        else:
            print("RANDOMLY INTIALSIING AS NO CHECKPOINT")

    def define_metrics(self, CFG):
        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.train_MSE = torchmetrics.MeanSquaredError()
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MSE = torchmetrics.MeanSquaredError()
        if CFG["use_localisation_head"]:
            self.test_loc_MSE = torchmetrics.MeanSquaredError()
            self.train_loc_MSE = torchmetrics.MeanSquaredError()

        self.train_err = torchmetrics.CatMetric()
        self.train_gt = torchmetrics.CatMetric()
        self.train_pred = torchmetrics.CatMetric()

        self.train_class_gt = torchmetrics.CatMetric()
        self.train_class_pred = torchmetrics.CatMetric()
        self.test_err = torchmetrics.CatMetric()
        self.test_gt = torchmetrics.CatMetric()
        self.test_pred = torchmetrics.CatMetric()

        self.test_class_gt = torchmetrics.CatMetric()
        self.test_class_pred = torchmetrics.CatMetric()

    def step(self, batch, batch_idx, tag):

        input, _boxes, gt_density, gt_cnt, im_id = batch
        gt_density = gt_density.squeeze(1)
        if self.CFG["split_up_img"] != 1:
            # put each split image through independantly (batch dimension)
            input = rearrange(input, "b n c h w -> (b n) c h w")
        feats = self.backbone(input)

        if self.CFG["use_localisation_head"]:

            density_prediction = self.localisation_head(feats)

            loss_localisation = self.loc_loss(density_prediction, gt_density)

            density_prediction_vis = density_prediction.clone()

            if self.CFG["save_visualise_localisation"] or (
                batch_idx == 0 and self.CFG["tensorboard_visualise_localisation"]
            ):
                self.visualise_localisation(
                    tag, input, gt_density, im_id, density_prediction_vis
                )

        if self.CFG["pca_visualise"]:
            self.visualise_features_pca(tag, input, im_id, feats)

        if self.CFG["use_counting_head"]:
            y_count, latent_head_feat = self.counting_head(feats)

            if self.CFG["split_up_img"] != 1:
                # combine them back into a single image
                y_count = rearrange(
                    y_count, "(b n) -> b n", n=self.CFG["split_up_img"] ** 2
                )
                gt_cnt = torch.sum(gt_cnt, dim=1)
                y_count = torch.sum(y_count, dim=1)

            if self.CFG["counting_loss"] == "MAE":
                loss_counting = torch.abs(y_count - gt_cnt).mean()

            elif self.CFG["counting_loss"] == "MSE":
                loss_counting = (torch.abs(y_count - gt_cnt) ** 2).mean()

            elif self.CFG["counting_loss"] == "MAPE":
                loss_counting = (torch.abs(y_count - gt_cnt) / gt_cnt).mean()

        if self.CFG["total_loss"] == "counting_loss":
            loss = loss_counting
        elif self.CFG["total_loss"] == "localisation_loss":
            loss = loss_localisation

        if self.CFG["dataset"] == "example_ims":
            for i in range(im_id.shape[0]):
                print(
                    im_id[i].item(),
                    gt_cnt[i].squeeze().item(),
                    y_count[i].squeeze().item(),
                )

        if batch_idx == 0 and self.CFG["tensorboard_visualise"]:

            self.tensorboard_vis(tag, input, gt_cnt, feats, y_count, latent_head_feat)

        self.log(
            f"{tag}/loss", torch.mean(loss), on_epoch=True, sync_dist=True,
        )

        if tag == "train":
            if self.CFG["use_counting_head"]:
                self.train_MAE(y_count.squeeze(), gt_cnt.squeeze())
                self.train_MSE(y_count.squeeze(), gt_cnt.squeeze())
                self.train_err(y_count.squeeze() - gt_cnt.squeeze())

                self.train_gt(gt_cnt)
                self.train_pred(y_count.squeeze())
            if self.CFG["use_localisation_head"]:
                self.train_loc_MSE(density_prediction, gt_density)

        elif tag == "val" or tag == "test":
            if self.CFG["use_counting_head"]:
                self.test_MAE(y_count.squeeze(), gt_cnt.squeeze())
                self.test_MSE(y_count.squeeze(), gt_cnt.squeeze())
                self.test_err(y_count.squeeze() - gt_cnt.squeeze())
                self.test_gt(gt_cnt)
                self.test_pred(y_count.squeeze())
            if self.CFG["use_localisation_head"]:
                self.test_loc_MSE(density_prediction, gt_density)

        return loss

    def tensorboard_vis(self, input, tag, gt_cnt, feats, y_count, latent_head_feat):
        if self.CFG["split_up_img"] != 1:
            # put each split image through independantly (batch dimension)
            input = rearrange(
                input,
                "(b n m) c h w -> b c (n h) (m w)",
                n=self.CFG["split_up_img"],
                m=self.CFG["split_up_img"],
            )
            input = self.resize224(input)

        if latent_head_feat != None:
            if len(latent_head_feat.shape) == 3:
                latent_head_feat = rearrange(
                    latent_head_feat,
                    "l b (h w) -> b l h w",
                    b=self.bs,
                    h=feats.shape[-2],
                    w=feats.shape[-1],
                )
            else:
                latent_head_feat = rearrange(
                    latent_head_feat,
                    "b (c h w) -> b c h w",
                    c=self.ch,
                    b=self.bs,
                    h=feats.shape[-2],
                    w=feats.shape[-1],
                )

            latent_head_feat = self.resize224(latent_head_feat)

        num_to_vis = min(4, self.bs)

        for i in range(num_to_vis):
            gt_i = np.asscalar(gt_cnt[i].cpu().numpy())
            y_i = np.asscalar(y_count[i].detach().cpu().numpy())

            img_i = denormalize(input[i].clone())
            img_i -= torch.min(img_i)
            img_i /= torch.max(img_i)
            strng = f"g:{gt_i:.1f}, p:{y_i:.1f}"

            if latent_head_feat != None:
                for j in range(latent_head_feat.shape[1]):
                    ii_i = latent_head_feat[i, j].squeeze()
                    ii_i -= torch.min(ii_i)
                    ii_i /= torch.max(ii_i)

                    vis_density_overlap = img_i / 2
                    vis_density_overlap[0] += ii_i / 2

                    vis_density_overlap = (
                        vis_density_overlap.permute(1, 2, 0).detach().cpu().numpy()
                    )
                    vis_density_overlap = (
                        (vis_density_overlap * 255).astype(np.uint8).copy()
                    )

                    vis_density_overlap = cv2.putText(
                        vis_density_overlap,
                        strng,
                        (0, int(30 * self.CFG["img_size"][0] / 384)),
                        cv2.FONT_HERSHEY_COMPLEX,
                        self.CFG["img_size"][0] / 384,
                        (255, 255, 255),
                        int(2 * self.CFG["img_size"][0] / 384),
                    )

                    self.logger.experiment.add_image(
                        f"{tag}/output_overlay_plt_{i}-{j}",
                        vis_density_overlap.transpose(2, 0, 1),
                        self.current_epoch,
                    )
            else:
                img_with_cnt = cv2.putText(
                    (img_i * 255)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    .copy(),
                    strng,
                    (0, int(30 * self.CFG["img_size"][0] / 384)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    self.CFG["img_size"][0] / 384,
                    (255, 255, 255),
                    int(2 * self.CFG["img_size"][0] / 384),
                )

                self.logger.experiment.add_image(
                    f"{tag}/output_{i}",
                    img_with_cnt.transpose(2, 0, 1),
                    self.current_epoch,
                )

    def visualise_features_pca(self, tag, input, im_id, feats):
        # pca on features
        feats_r = self.resize224b(feats)
        b, _c, h, w = feats_r.shape
        feats_r_stacked = rearrange(feats_r, "b c h w -> (b h w) c")
        feats_stacked = rearrange(feats, "b c h w -> (b h w) c")

        pca = PCA(n_components=1)
        pca.fit(feats_stacked.detach().cpu().numpy())
        x = pca.transform(feats_r_stacked.detach().cpu().numpy())
        x = rearrange(x, "(b h w) c -> b c h w", c=1, b=b, h=h, w=w)

        for i, (inp, fet) in enumerate(zip(input, x)):
            img_i = denormalize(inp)
            img_i -= torch.min(img_i)
            img_i /= torch.max(img_i)
            img_i = img_i.detach().cpu().numpy()

            img_i = np.mean(img_i, axis=0)
            img_i = np.stack((img_i, img_i, img_i), axis=0)

            for ch in range(3):
                img_i[ch] = img_i[ch] / 2.0

            fet = fet[0]
            fet -= np.min(fet)
            fet /= np.max(fet)

            cm = plt.get_cmap("gnuplot")
            fet = fet ** 2
            colored_image = cm(fet)[:, :, :3]
            img_i = img_i / 2 + colored_image.transpose(2, 0, 1) / 2

            if self.CFG["save_visualise_localisation"]:
                img_i = (img_i.transpose(1, 2, 0) * 255).astype(np.uint8)
                im = Image.fromarray(img_i)
                im.save(f"output/pca/{tag}_{str(im_id[i].item())}_pca.jpg")

    def visualise_localisation(
        self, tag, input, gt_density, im_id, density_prediction_vis
    ):
        for i in range(self.bs):
            den = density_prediction_vis[i].clone()
            inp = input[i].clone()

            den -= torch.min(den)
            den /= torch.max(den)
            den = den.detach().cpu().numpy().transpose(1, 2, 0)

            img_i = denormalize(inp)
            img_i -= torch.min(img_i)
            img_i /= torch.max(img_i)
            img_i = img_i.detach().cpu().numpy()

            img_i = np.mean(img_i, axis=0)
            img_i = np.stack((img_i, img_i, img_i), axis=2)

            for ch in range(3):
                img_i[ch] = img_i[ch] / 2.0

            cm = plt.get_cmap("jet")

            den = cm(den.squeeze())[:, :, :3]

            den = den / 2 + img_i / 2

            den_im = (den * 255).astype(np.uint8)
            im = Image.fromarray(den_im)
            im.save(f"output/localisation/{tag}_{str(im_id[i].item())}_loc.jpg")

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.CFG["test_split"])

    def training_epoch_end(self, _outputs):
        if self.CFG["use_counting_head"]:
            tr_mae = self.train_MAE.compute()
            tr_mse = self.train_MSE.compute()
            tr_gt = self.train_gt.compute()
            tr_preds = self.train_pred.compute()
            tr_err = self.train_err.compute()

            if tr_mae.get_device() == 0:
                # only print it once per epoch not once per gpu
                print(
                    f" trn over all GPUS, MAE: {tr_mae:.2f}, RMSE: {torch.sqrt(tr_mse):.2f}"
                )
                self.logger.experiment.add_histogram(
                    "train/err", tr_err, self.current_epoch
                )
                self.logger.experiment.add_histogram(
                    "train/gt", tr_gt, self.current_epoch
                )
                self.logger.experiment.add_histogram(
                    "train/preds", tr_preds, self.current_epoch
                )

            self.logger.experiment.add_scalar(
                "train/DDP_MAE", tr_mae, global_step=self.current_epoch,
            )
            self.logger.experiment.add_scalar(
                "train/DDP_RMSE", torch.sqrt(tr_mse), global_step=self.current_epoch,
            )

        if self.CFG["use_localisation_head"]:
            train_loc_mse = self.train_loc_MSE.compute()
            self.log(
                "train/Loc_MSE", train_loc_mse,
            )
            self.test_loc_MSE.reset()

        self.train_MAE.reset()
        self.train_MSE.reset()
        self.train_gt.reset()
        self.train_err.reset()
        self.train_pred.reset()

    def validation_epoch_end(self, outputs):
        if self.CFG["use_counting_head"]:
            test_mae = self.test_MAE.compute()
            test_mse = self.test_MSE.compute()
            test_err = self.test_err.compute()
            test_gt = self.test_gt.compute()
            test_preds = self.test_pred.compute()

            if test_mae.get_device() == 0:

                print(
                    f"   {self.CFG['test_split']} over all GPUS, MAE: {test_mae:.2f}, RMSE: {torch.sqrt(test_mse):.2f}"
                )

                self.logger.experiment.add_histogram(
                    f"{self.CFG['test_split']}/gt", test_gt, self.current_epoch
                )
                self.logger.experiment.add_histogram(
                    f"{self.CFG['test_split']}/err", test_err, self.current_epoch
                )
                self.logger.experiment.add_histogram(
                    f"{self.CFG['test_split']}/preds", test_preds, self.current_epoch
                )

            self.log(
                f"{self.CFG['test_split']}_DDP_MAE", test_mae,
            )
            self.log(
                f"{self.CFG['test_split']}_DDP_RMSE", torch.sqrt(test_mse),
            )

        if self.CFG["use_localisation_head"]:
            test_loc_mse = self.test_loc_MSE.compute()
            self.log(
                f"{self.CFG['test_split']}_DDP_Loc_MSE", test_loc_mse,
            )
            self.test_loc_MSE.reset()

        self.test_MAE.reset()
        self.test_MSE.reset()
        self.test_err.reset()
        self.test_gt.reset()
        self.test_pred.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.CFG["learning_rate"]),
            weight_decay=self.CFG["weight_decay"],
        )

        if self.CFG["scheduler"] == "LinearWarmupCosineAnneal":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.CFG["warmup_epochs"],
                max_epochs=self.CFG["max_epochs"],
            )
            scheduler = {"scheduler": scheduler}
            return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

        elif self.CFG["scheduler"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.CFG["plateau_factor"],
                patience=self.CFG["plateau_patience"],
            )
            scheduler = {"scheduler": scheduler, "monitor": "val_DDP_MAE"}
            return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

        elif self.CFG["scheduler"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=self.CFG["scheduler_steps"],
                gamma=self.CFG["scheduler_gamma"],
            )
            return ({"optimizer": optimizer, "lr_scheduler": scheduler},)

        return optimizer

    def load_pretrained_model(self):
        model_dict = self.state_dict()

        updated_pretrained_dict = {}

        if self.CFG["resume_path"] != "":
            pretrained_dict_path = self.CFG["resume_path"]
            print("loading model checkpoint: ", pretrained_dict_path)
            pretrained_dict = torch.load(pretrained_dict_path)["state_dict"]
            for k, v in pretrained_dict.items():
                updated_pretrained_dict[k] = v

        pretrained_dict = updated_pretrained_dict

        if self.CFG["resume_counting_head_path"] != "":
            counting_head_pretrained_dict_path = self.CFG["resume_counting_head_path"]
            counting_head_pretrained_dict = torch.load(
                counting_head_pretrained_dict_path
            )["state_dict"]
            print(
                "loading counting head checkpoint: ",
                self.CFG["resume_counting_head_path"],
            )

            counting_head_pretrained_dict = {
                k: v
                for k, v in counting_head_pretrained_dict.items()
                if "counting_head." in k
            }

            pretrained_dict.update(counting_head_pretrained_dict)

        if self.CFG["resume_path_localisation"] != "":
            localisation_head_pretrained_dict_path = self.CFG[
                "resume_path_localisation"
            ]
            localisation_head_pretrained_dict = torch.load(
                localisation_head_pretrained_dict_path
            )["state_dict"]
            print(
                "loading localisation head checkpoint: ",
                self.CFG["resume_path_localisation"],
            )

            localisation_head_pretrained_dict = {
                k: v
                for k, v in localisation_head_pretrained_dict.items()
                if "localisation_head." in k
            }

            pretrained_dict.update(localisation_head_pretrained_dict)

        if pretrained_dict.keys() != model_dict.keys():
            print("LOADING MODEL AND CREATED MODEL NOT THE SAME")

        print(
            f"model_dict: {len(model_dict.keys())}, pretrained_dict: {len(pretrained_dict.keys())}"
        )

        p_kys = pretrained_dict.keys()
        m_kys = model_dict.keys()
        in_p_not_m = list(set(p_kys) - set(m_kys))
        in_m_not_p = list(set(m_kys) - set(p_kys))

        if len(in_p_not_m) > 0:
            print(f"LAYERS IN THE CHECKPOINT BUT NOT IN THE MODEL: {in_p_not_m}")

        if len(in_m_not_p) > 0:
            print(f"LAYERS IN THE MODEL BUT NOT IN THE CHECKPOINT: {in_m_not_p}")

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)

    def val_dataloader(self):
        dataloader = get_dataloader(self.CFG, train=False)
        return dataloader

    def train_dataloader(self):
        dataloader = get_dataloader(self.CFG, train=True)
        return dataloader
