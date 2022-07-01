import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import Block

from utils.data_utils import denormalize
from utils.vis_utils import get_layer_maps

from models.backbone_convnext import convnext_base
from models.backbone_vit import ViTExtractor
from models.counting_head import CountingHead


class CountingAnything(LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.train_MSE = torchmetrics.MeanSquaredError()
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MSE = torchmetrics.MeanSquaredError()
        resloution = 28

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

            self.feat_extractor = ViTExtractor(vit_config)
        elif CFG["backbone"] == "resnet":
            # resnet-50
            feature_dim = 512
            backbone = models.resnet50(pretrained=True)
            layers = list(backbone.children())[:-4]
            self.feat_extractor = nn.Sequential(*layers)

        elif CFG["backbone"] == "convnext":  #
            # convnext_base
            feature_dim = 256
            self.feat_extractor = convnext_base(pretrained=True)

        for param in self.feat_extractor.parameters():
            param.requires_grad = False

        if "simple" in self.CFG["count_head_type"]:
            complexity = "simple"
            self.ch = 3
        elif "complex" in self.CFG["count_head_type"]:
            complexity = "complex"
            self.ch = 3

        self.counting_head = CountingHead(
            feature_dim, resloution, c=self.ch, complexity=complexity
        )

        if self.CFG["resume_path"] != "" or (
            self.CFG["counting_head_path"] != ""
            and self.CFG["only_resume_counting_head"]
        ):
            self.load_pretrained_model()
        else:
            print("RANDOMLY INTIALSIING AS NO CHECKPOINT")

        if self.CFG["tensorboard_visualise"] or self.CFG["save_ims"]:
            self.resize_im_for_visualisation = T.Resize(
                self.CFG["img_size"][0], T.InterpolationMode.NEAREST
            )

    def forward(self, x):
        return self.counting_head(x)

    def step(self, batch, batch_idx, tag):

        input, _boxes, _gt_density, gt_cnt, im_id = batch

        feats = self.feat_extractor(input)
        y_count, intermediate_image = self(feats)

        if self.CFG["loss"] == "MAE":
            loss = torch.abs(y_count - gt_cnt).mean()
        elif self.CFG["loss"] == "MSE":
            loss = (torch.abs(y_count - gt_cnt) ** 2).mean()

        if (batch_idx == 0 and self.CFG["tensorboard_visualise"]) or self.CFG[
            "save_ims"
        ]:
            if intermediate_image != None:
                if len(intermediate_image.shape) == 3:
                    intermediate_image = rearrange(
                        intermediate_image,
                        "l b (h w) -> b l h w",
                        b=feats.shape[0],
                        h=feats.shape[-2],
                        w=feats.shape[-1],
                    )
                else:
                    intermediate_image = rearrange(
                        intermediate_image,
                        "b (c h w) -> b c h w",
                        c=self.ch,
                        b=feats.shape[0],
                        h=feats.shape[-2],
                        w=feats.shape[-1],
                    )

                intermediate_image = self.resize_im_for_visualisation(
                    intermediate_image
                )

                num_to_vis = (
                    input.shape[0] if self.CFG["save_ims"] else min(4, input.shape[0])
                )

                for i in range(num_to_vis):
                    gt_i = np.asscalar(gt_cnt[i].cpu().numpy())
                    y_i = np.asscalar(y_count[i].detach().cpu().numpy())

                    if (
                        intermediate_image.shape[1] <= 3
                        and intermediate_image.shape[1] > 1
                    ):
                        vis_density_overlap = get_layer_maps(
                            i,
                            self.CFG["img_size"][0],
                            intermediate_image,
                            input,
                        )

                        if self.CFG["save_ims"]:
                            save_path = self.CFG[
                                "output_dir"
                            ] + "/img_id_{}_gt_{}_pred_{:.2f}.png".format(
                                im_id[i],
                                gt_i,
                                y_i,
                            )

                            img = Image.fromarray(vis_density_overlap, "RGB")
                            img.save(save_path)

                        self.logger.experiment.add_image(
                            f"{tag}/linear_output_plt_{i}",
                            vis_density_overlap.transpose(2, 0, 1),
                            self.current_epoch,
                        )

        self.log(
            f"{tag}/loss",
            torch.mean(loss),
            on_epoch=True,
            sync_dist=True,
        )

        if tag == "train":
            self.train_MAE(y_count, gt_cnt)
            self.train_MSE(y_count, gt_cnt)
        elif tag == "val" or tag == "test":
            self.test_MAE(y_count, gt_cnt)
            self.test_MSE(y_count, gt_cnt)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.CFG["test_split"])

    def training_epoch_end(self, outputs):
        tr_mae = self.train_MAE.compute()
        tr_mse = self.train_MSE.compute()

        self.logger.experiment.add_scalar(
            "train/DDP_MAE",
            tr_mae,
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            "train/DDP_RMSE",
            torch.sqrt(tr_mse),
            global_step=self.current_epoch,
        )
        self.train_MAE.reset()
        self.train_MSE.reset()

    def validation_epoch_end(self, outputs):
        test_mae = self.test_MAE.compute()
        test_mse = self.test_MSE.compute()
        if test_mae.get_device() == 0:
            # only print it once per epoch not once per device
            print(
                f" {self.CFG['test_split']} over all GPUS, MAE: {test_mae}, RMSE: {torch.sqrt(test_mse)}"
            )

        self.log(
            self.CFG["test_split"] + "_DDP_MAE",
            test_mae,
        )
        self.log(
            self.CFG["test_split"] + "_DDP_RMSE",
            torch.sqrt(test_mse),
        )
        self.test_MAE.reset()
        self.test_MSE.reset()

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

        return optimizer

    def on_save_checkpoint(self, b):
        if self.CFG["save_counting_head_seperately"]:
            dirpath = list(b["callbacks"].values())[1]["dirpath"]
            torch.save(
                {
                    "state_dict": self.counting_head.state_dict(),
                },
                dirpath + f"/counting_head_epoch={self.current_epoch}.ckpt",
            )

    def load_pretrained_model(self):
        model_dict = self.state_dict()

        if self.CFG["only_resume_counting_head"]:

            model_dict = {k: v for k, v in model_dict.items() if "counting_head" in k}

            pretrained_dict_path = self.CFG["counting_head_path"]
            pretrained_dict = torch.load(pretrained_dict_path)["state_dict"]
            print("loading head checkpoint: ", self.CFG["counting_head_path"])

            pretrained_dict = {
                "counting_head." + k: v for k, v in pretrained_dict.items()
            }

        else:
            pretrained_dict_path = self.CFG["resume_path"]
            pretrained_dict = torch.load(pretrained_dict_path)["state_dict"]

            print("loading model checkpoint: ", self.CFG["resume_path"])

            # filter out unnecessary keys
            if pretrained_dict.keys() != model_dict.keys():
                print("LOADING MODEL AND CREATED MODEL NOT THE SAME")

        print("model_dict", len(model_dict.keys()))
        print("pretrained_dict", len(pretrained_dict.keys()))

        p_kys = pretrained_dict.keys()
        m_kys = model_dict.keys()
        in_p_not_m = list(set(p_kys) - set(m_kys))
        in_m_not_p = list(set(m_kys) - set(p_kys))

        if len(in_p_not_m) > 0:
            print("LAYERS IN THE CHECKPOINT BUT NOT IN THE MODEL", in_p_not_m)

        if len(in_m_not_p) > 0:
            print("LAYERS IN THE MODEL BUT NOT IN THE CHECKPOINT", in_m_not_p)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
        self.load_state_dict(pretrained_dict, strict=False)
