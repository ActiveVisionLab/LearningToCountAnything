import logging
import torch

from torchvision import transforms
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import random
import torchvision.transforms.functional as trans_F
import torchvision.transforms as T

from scipy import io
import json
from PIL import Image
import numpy as np
import cv2
import os


logger = logging.getLogger(__name__)

MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


Normalize_PIL = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)]
)
Normalize_tensor = transforms.Compose(
    [transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)]
)


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD, clip_0_1=True):
    # Reverses the normalisation on a tensor.
    with torch.no_grad():
        denormalized = tensor.clone()

        for channel, mean, std in zip(denormalized, means, stds):
            channel.mul_(std).add_(mean)

            if clip_0_1:
                channel[channel < 0] = 0
                channel[channel > 1] = 1

        return denormalized


class CountingDatasetFSC(Dataset):
    def __init__(self, CFG, train, transform):
        self.CFG = CFG
        self.img_size = CFG["img_size"]
        self.train = train
        self.img_channels = CFG["img_channels"]
        self.dataset = CFG["dataset"]

        if CFG["dataset"] == "FSC-147":
            data_path = CFG["data_path"] + "FSC-147/"
            anno_file = data_path + "annotation_FSC147_384.json"
            data_split_file = data_path + "Train_Test_Val_FSC_147.json"
        elif CFG["dataset"] == "FSC-133":
            data_path = CFG["data_path"] + "FSC-133/"
            anno_file = data_path + "annotation_FSC133_384.json"
            data_split_file = data_path + "Train_Test_Val_FSC_133.json"
        self.im_dir = data_path + "images_384_VarV2"
        self.gt_dir = data_path + "gt_density_map_adaptive_384_VarV2"

        if anno_file != None:
            with open(anno_file) as f:
                self.annotations = json.load(f)
        else:
            self.annotations = None

        if data_split_file != None:
            with open(data_split_file) as f:
                data_split = json.load(f)
            if self.train:
                self.im_ids = data_split["train"]

            else:
                self.im_ids = data_split[CFG["test_split"]]

        else:
            self.im_ids = [
                f
                for f in os.listdir(self.im_dir)
                if os.path.isfile(self.im_dir + "/" + f)
            ]

        # exclude images where the count is above 500 or 1000
        if (
            CFG["exclude_imgs_with_counts_over_500"]
            or CFG["exclude_imgs_with_counts_over_1000"]
        ):
            self.remove_im_ids_if_high_density(CFG)

        # exclude images where the average bbox size or aspect ratio are in the extremes
        if (
            CFG["bboxes_sizes_to_look_at"] != "all"
            or CFG["bboxes_aspect_ratio_to_look_at"] != "all"
        ):
            self.remove_im_ids_based_on_bbox_constraints(CFG)

        splt = "train" if train else CFG["test_split"]
        print(f"{splt} set, size:{len(self.im_ids)}, eg: {self.im_ids[:5]}")

        self.transform = transform

    def remove_im_ids_if_high_density(self, CFG):
        # remove image ids from list if there are over 500 or over 1000 instances
        over_500s, over_1000s = FSC147_over_500_over_1000()

        how_many_remove_1000 = 0
        removed_1000 = []
        for idx in over_1000s:
            if idx in self.im_ids:
                how_many_remove_1000 += 1
                removed_1000.append(idx)
                self.im_ids.remove(idx)
        print(
            f"{how_many_remove_1000} images excluded for having a count of more than 1000",
            removed_1000,
        )
        if CFG["exclude_imgs_with_counts_over_500"]:
            how_many_remove_500 = 0
            for idx in over_500s:
                if idx in self.im_ids:
                    how_many_remove_500 += 1
                    self.im_ids.remove(idx)
            print(
                f"{how_many_remove_500} images excluded for having a count of more than 500",
            )

    def remove_im_ids_based_on_bbox_constraints(self, CFG):
        # remove im ids from list if extreme sizes and/or aspect ratios
        selective_ids = []
        all_bboxes_size = []
        all_bboxes_aspect_ratio = []
        selective_ids = []
        for id in self.im_ids:
            anno = self.annotations[id]
            h = anno["H"] * anno["ratio_h"]
            w = anno["W"] * anno["ratio_w"]
            bboxes = anno["box_examples_coordinates"]
            bbox_area_s = []
            aspect_ratio_ok = True

            # find the area and aspect ratio of all given bounding boxes on a sinlge image
            for bbox in bboxes:
                # find area of each bbox
                x1 = bbox[2][0] - bbox[0][0]  # x
                x2 = bbox[2][1] - bbox[0][1]  # y
                norm_area = x1 * x2 / (h * w)
                aspect_ratio = x1 / x2
                if aspect_ratio > 1:
                    all_bboxes_aspect_ratio.append(aspect_ratio)
                else:
                    all_bboxes_aspect_ratio.append(1 / aspect_ratio)

                if CFG["bboxes_aspect_ratio_to_look_at"] == "square" and (
                    aspect_ratio > CFG["bbox_aspect_ratio_max"]
                    or aspect_ratio < 1 / CFG["bbox_aspect_ratio_max"]
                ):
                    # if we care about aspect ratio and its bad
                    aspect_ratio_ok = False

                bbox_area_s.append(norm_area)

            mean_area = sum(bbox_area_s) / len(bbox_area_s)
            all_bboxes_size.append(mean_area)

            # remove images where the average bounding box size is not in the set bounds
            if CFG["bboxes_sizes_to_look_at"] == "medium":
                if (
                    CFG["bbox_small_area"] <= mean_area
                    and mean_area <= CFG["bbox_big_area"]
                    and aspect_ratio_ok
                ):
                    selective_ids.append(id)

            elif CFG["bboxes_sizes_to_look_at"] == "small":
                if mean_area <= CFG["bbox_small_area"] and aspect_ratio_ok:
                    selective_ids.append(id)
            elif CFG["bboxes_sizes_to_look_at"] == "big":
                if mean_area >= CFG["bbox_big_area"] and aspect_ratio_ok:
                    selective_ids.append(id)
            else:
                print("bboxes_sizes_to_look_at needs to be all, small, medium, or big")
                exit()

        print(
            f"{len(selective_ids)}/{len(self.im_ids)} images remaining after removing size and/or aspect ratio outliers"
        )

        self.im_ids = selective_ids
        print(f"selective_ids: {len(selective_ids)}")

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):

        im_id = self.im_ids[idx]
        image = Image.open(f"{self.im_dir}/{im_id}")
        image.load()
        if image.mode != "RGB":
            print("IMAGE NOT RGB", im_id)
        rects = list()
        if self.annotations != None:
            anno = self.annotations[im_id]
            bboxes = anno["box_examples_coordinates"]
            dots = np.array(anno["points"])

            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            rects = rects[:3]
            density_path = self.gt_dir + "/" + im_id.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype("float32")

        else:
            print("no annotations")
            dots = None
            density = np.zeros((image.size[1], image.size[0]))
            if "count" in im_id:
                c = im_id.split("count")[1]
                print(im_id, c)
                density[0, 0] = c

        sample = {
            "image": image,
            "lines_boxes": rects,
            "gt_density": density,
            "dots": dots,
        }

        sample = self.transform(sample)

        if self.img_channels == 1:
            sample["image"] = torch.mean(sample["image"], dim=0).unsqueeze(0)

        img_id = torch.tensor(int(im_id.split(".jpg")[0]))
        return (
            sample["image"],
            sample["boxes"],
            sample["gt_density"],
            sample["count"],
            img_id,
        )


class CountingDatasetCARPK(Dataset):
    def __init__(self, CFG, train, transform):
        self.CFG = CFG
        self.img_size = CFG["img_size"]
        self.train = train
        self.img_channels = CFG["img_channels"]
        self.dataset = CFG["dataset"]

        data_path = CFG["data_path"] + "CARPK/datasets/CARPK_devkit/data/"
        self.anno_dir = data_path + "Annotations"
        self.im_dir = data_path + "Images"

        split_path = data_path + "ImageSets/"
        if self.train:
            split_path += "train.txt"
        else:
            split_path += "test.txt"

        my_file = open(split_path, "r")
        data = my_file.read()
        self.im_ids = data.split("\n")[:-1]
        my_file.close()

        splt = "train" if train else "test"
        print(f"{splt} set, size:{len(self.im_ids)}, eg: {self.im_ids[:5]}")

        self.transform = transform

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):

        im_id = self.im_ids[idx]
        image = Image.open(f"{self.im_dir}/{im_id}.png")
        image.load()
        if image.mode != "RGB":
            print("IMAGE NOT RGB", im_id)
        rects = list()

        my_file = open(self.anno_dir + "/" + im_id + ".txt", "r")
        data = my_file.read()
        boxes = data.split("\n")[:-1]
        gt_cnt = len(boxes)
        my_file.close()

        dots = None
        dots = np.zeros((gt_cnt, 2))
        density = np.zeros((image.size[1], image.size[0]))

        sample = {
            "image": image,
            "lines_boxes": rects,
            "gt_density": density,
            "dots": dots,
        }

        sample = self.transform(sample)

        if self.img_channels == 1:
            sample["image"] = torch.mean(sample["image"], dim=0).unsqueeze(0)

        img_id = torch.tensor(int(im_id.split("_")[0] + im_id.split("_")[2]))
        return (
            sample["image"],
            sample["boxes"],
            sample["gt_density"],
            sample["count"],
            img_id,
        )


class CountingDatasetShanghai(Dataset):
    def __init__(self, CFG, train, transform):
        self.CFG = CFG
        self.img_size = CFG["img_size"]
        self.train = train
        self.img_channels = CFG["img_channels"]
        self.dataset = CFG["dataset"]
        self.shanghai_part = self.dataset.split("_")[1]

        if self.shanghai_part == "A":
            data_path = CFG["data_path"] + "ShanghaiTech/part_A/"
        elif self.shanghai_part == "B":
            data_path = CFG["data_path"] + "ShanghaiTech/part_B/"

        self.im_dir = data_path + "images"
        self.gt_dir = data_path + "ground-truth"

        self.annotations = None

        if self.train:
            self.im_dir = data_path + "train_data/" "images"
            self.gt_dir = data_path + "train_data/" "ground-truth"
        else:
            self.im_dir = data_path + "test_data/" + "images"
            self.gt_dir = data_path + "test_data/" + "ground-truth"

        self.im_ids = [
            f for f in os.listdir(self.im_dir) if os.path.isfile(self.im_dir + "/" + f)
        ]

        splt = "train" if train else "test"
        print(f"{splt} set, size:{len(self.im_ids)}, eg: {self.im_ids[:5]}")

        self.transform = transform

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):

        im_id = self.im_ids[idx]
        image = Image.open(f"{self.im_dir}/{im_id}")
        image.load()

        mat = io.loadmat(f"{self.gt_dir}/GT_{im_id.split('.')[0]}.mat")
        im_infor = mat["image_info"]
        dots = im_infor[0][0][0][0][0]
        density = np.zeros((image.size[1], image.size[0]))
        if image.mode != "RGB":
            print("IMAGE NOT RGB", im_id, image.mode)
            image = image.convert("RGB")

        rects = list()
        sample = {
            "image": image,
            "lines_boxes": rects,
            "gt_density": density,
            "dots": dots,
        }
        sample = self.transform(sample)

        if self.img_channels == 1:
            sample["image"] = torch.mean(sample["image"], dim=0).unsqueeze(0)

        img_id = torch.tensor(int(im_id.split("_")[1].split(".jpg")[0]))

        return (
            sample["image"],
            sample["boxes"],
            sample["gt_density"],
            sample["count"],
            img_id,
        )


class ExampleImagesDataset(Dataset):
    def __init__(self, CFG, train):
        # used to run small numbers of images from a seperate directory

        self.img_size = CFG["img_size"]
        self.train = train
        self.im_dir = "/data/example_ims"
        self.im_ids = [
            f for f in os.listdir(self.im_dir) if os.path.isfile(self.im_dir + "/" + f)
        ]

        splt = "train" if train else "test"
        print(f"{splt} set, size:{len(self.im_ids)}, eg: {self.im_ids[:5]}")
        self.trans = transforms.Compose(
            [
                transforms.Resize((self.img_size[0], self.img_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD),
            ]
        )

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):

        im_id = self.im_ids[idx]
        image = Image.open(f"{self.im_dir}/{im_id}")
        image.load()
        if image.mode != "RGB":
            print("IMAGE NOT RGB", im_id, image.mode)
            image = image.convert("RGB")
        image = self.trans(image)

        fl_name = im_id.split(".")[0]
        img_id, gt_cnt = fl_name.split("_")
        gt_cnt = torch.Tensor([int(gt_cnt)])
        img_id = torch.tensor(int(img_id))
        density = torch.zeros((1, 1, image.shape[1], image.shape[2]))
        _boxes = torch.Tensor((1))

        return (
            image,
            _boxes,
            density,
            gt_cnt,
            img_id,
        )


def get_dataloader(CFG, train):
    TransformT = transforms.Compose([resize_instance(CFG, train=train)])

    if CFG["dataset"] == "example_ims":
        dataset = ExampleImagesDataset(CFG, train=train, transform=TransformT)
    elif CFG["dataset"] == "Shanghai_A" or CFG["dataset"] == "Shanghai_B":
        dataset = CountingDatasetShanghai(CFG, train=train, transform=TransformT)
    elif CFG["dataset"] == "carpk":
        dataset = CountingDatasetCARPK(CFG, train=train, transform=TransformT)
    else:
        dataset = CountingDatasetFSC(CFG, train=train, transform=TransformT)

    if train:
        sampler = RandomSampler(dataset)
        bs = CFG["train_batch_size"]
    else:
        sampler = SequentialSampler(dataset)
        bs = CFG["eval_batch_size"]

    train_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=CFG["drop_last"],
    )

    return train_loader


def FSC147_over_500_over_1000():
    # return lists of images with over 500 and over 1000 count
    # all images in train val and test set with 500<count<1000
    over_500s = [
        "3136.jpg",
        "3356.jpg",
        "2342.jpg",
        "4128.jpg",
        "6328.jpg",
        "2390.jpg",
        "4341.jpg",
        "2348.jpg",
        "880.jpg",
        "830.jpg",
        "840.jpg",
        "957.jpg",
        "975.jpg",
        "1915.jpg",
        "1936.jpg",
        "3425.jpg",
        "3437.jpg",
        "3433.jpg",
        "3665.jpg",
        "5860.jpg",
        "6969.jpg",
        "687.jpg",
        "2159.jpg",
        "2243.jpg",
        "6281.jpg",
        "6644.jpg",
        "7473.jpg",
        "6860.jpg",
    ]
    # all images in train val and test set with 1000<count
    over_1000s = [
        "7603.jpg",
        "805.jpg",
        "2360.jpg",
        "804.jpg",
        "2775.jpg",
        "2347.jpg",
        "865.jpg",
        "949.jpg",
        "935.jpg",
        "1956.jpg",
        "7656.jpg",
        "1123.jpg",
        "7611.jpg",
    ]
    return over_500s, over_1000s


class resize_instance(object):
    def __init__(self, CFG, train):
        self.img_size = CFG["img_size"]
        self.output_h = CFG["img_size"][0]
        self.output_w = CFG["img_size"][1]
        self.img_mode = CFG["img_mode"]
        self.trans = CFG["image_transforms"]
        self.split_up_img = CFG["split_up_img"]

        # if we are later going to split up the image then we make it bigger now so the splits are the right size
        self.output_h = self.split_up_img * self.output_h
        self.output_w = self.split_up_img * self.output_w

        if "increase_density" in self.trans:
            self.increase_density_ratio = CFG["increase_density_ratio"]
            self.increase_density_amount = CFG["increase_density_amount"]
        self.train = train

    def __call__(self, sample):
        image, lines_boxes, density, dots = (
            sample["image"],
            sample["lines_boxes"],
            sample["gt_density"],
            sample["dots"],
        )
        W, H = image.size

        scale_factor_h = self.output_h / H
        scale_factor_w = self.output_w / W
        new_H = self.output_h
        new_W = self.output_w

        if self.img_mode == "scale" or scale_factor_h == scale_factor_w:
            resized_image = transforms.Resize((new_H, new_W))(image)

            if isinstance(density, torch.Tensor):
                resized_density = transforms.Resize(
                    (resized_image.size[1], resized_image.size[0])
                )(density)

            else:
                resized_density = cv2.resize(
                    density, (resized_image.size[0], resized_image.size[1])
                )
                density = torch.from_numpy(density)
                resized_density = (
                    torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
                )
            orig_count = torch.sum(density)
            new_count = torch.sum(resized_density)

            if new_count > 0:
                resized_density = resized_density * (orig_count / new_count)

            boxes = list()
            for i, box in enumerate(lines_boxes):
                y1, x1, y2, x2 = (
                    box[0] * scale_factor_h,
                    box[1] * scale_factor_w,
                    box[2] * scale_factor_h,
                    box[3] * scale_factor_w,
                )
                boxes.append([0, y1, x1, y2, x2])

            boxes = torch.Tensor(boxes).unsqueeze(0)

            if dots is not None:
                dots_new = torch.zeros_like(torch.from_numpy(dots))
                for i, dot in enumerate(dots):
                    dots_new[i, 0] = dot[0] * scale_factor_w
                    dots_new[i, 1] = dot[1] * scale_factor_h
            else:
                dots_new = [[], []]
                dots_new = torch.tensor(dots_new).permute(1, 0)

        resized_image = Normalize_PIL(resized_image)
        if self.train:
            if "increase_density" in self.trans and random.random() > (
                self.increase_density_ratio / 100
            ):
                if self.increase_density_amount == 16 and random.random() > 0.5:
                    tile_num = 16
                else:
                    tile_num = 4

                if tile_num == 4:
                    reduced_size = transforms.Resize(
                        [int(self.output_h / 2), int(self.output_w / 2)]
                    )
                elif tile_num == 16:
                    reduced_size = transforms.Resize(
                        [int(self.output_h / 4), int(self.output_w / 4)]
                    )

                resized_image = reduced_size(resized_image)
                resized_density = reduced_size(resized_density)
                smaller_ims = []
                smaller_densities = []
                for i in range(tile_num):
                    resized_image_i = resized_image.clone()
                    resized_density_i = resized_density.clone()
                    if "reflect_rotate" in self.trans:
                        if random.random() > 0.5:
                            resized_image_i = trans_F.hflip(resized_image_i)
                            resized_density_i = trans_F.hflip(resized_density_i)
                        rotate_angle = int(random.random() * 4)
                        if rotate_angle != 0:
                            resized_image_i = trans_F.rotate(
                                resized_image_i, rotate_angle * 90
                            )
                            resized_density_i = trans_F.rotate(
                                resized_density_i, rotate_angle * 90
                            )

                    smaller_ims.append(resized_image_i)
                    smaller_densities.append(resized_density_i)

                if tile_num == 4:
                    resized_image = torch.cat(
                        (
                            torch.cat((smaller_ims[0], smaller_ims[1]), dim=1),
                            torch.cat((smaller_ims[2], smaller_ims[3]), dim=1),
                        ),
                        dim=2,
                    )
                    resized_density = torch.cat(
                        (
                            torch.cat(
                                (smaller_densities[0], smaller_densities[1]), dim=2
                            ),
                            torch.cat(
                                (smaller_densities[2], smaller_densities[3]), dim=2
                            ),
                        ),
                        dim=3,
                    )

                    dots_new = torch.cat(
                        (dots_new, dots_new, dots_new, dots_new), dim=0
                    )
                elif tile_num == 16:
                    resized_image = torch.cat(
                        (
                            torch.cat(
                                (
                                    smaller_ims[0],
                                    smaller_ims[1],
                                    smaller_ims[2],
                                    smaller_ims[3],
                                ),
                                dim=1,
                            ),
                            torch.cat(
                                (
                                    smaller_ims[4],
                                    smaller_ims[5],
                                    smaller_ims[6],
                                    smaller_ims[7],
                                ),
                                dim=1,
                            ),
                            torch.cat(
                                (
                                    smaller_ims[8],
                                    smaller_ims[9],
                                    smaller_ims[10],
                                    smaller_ims[11],
                                ),
                                dim=1,
                            ),
                            torch.cat(
                                (
                                    smaller_ims[12],
                                    smaller_ims[13],
                                    smaller_ims[14],
                                    smaller_ims[15],
                                ),
                                dim=1,
                            ),
                        ),
                        dim=2,
                    )
                    resized_density = torch.cat(
                        (
                            torch.cat(
                                (
                                    smaller_densities[0],
                                    smaller_densities[1],
                                    smaller_densities[2],
                                    smaller_densities[3],
                                ),
                                dim=2,
                            ),
                            torch.cat(
                                (
                                    smaller_densities[4],
                                    smaller_densities[5],
                                    smaller_densities[6],
                                    smaller_densities[7],
                                ),
                                dim=2,
                            ),
                            torch.cat(
                                (
                                    smaller_densities[8],
                                    smaller_densities[9],
                                    smaller_densities[10],
                                    smaller_densities[11],
                                ),
                                dim=2,
                            ),
                            torch.cat(
                                (
                                    smaller_densities[12],
                                    smaller_densities[13],
                                    smaller_densities[14],
                                    smaller_densities[15],
                                ),
                                dim=2,
                            ),
                        ),
                        dim=3,
                    )

                    dots_new = torch.cat(
                        (
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                            dots_new,
                        ),
                        dim=0,
                    )

            if "reflect_rotate" in self.trans:
                if random.random() > 0.5:
                    resized_image = trans_F.hflip(resized_image)
                    resized_density = trans_F.hflip(resized_density)
                    dots_new[:, 0] = self.output_w - dots_new[:, 0]

                rotate_angle = int(random.random() * 4)
                if rotate_angle != 0:
                    resized_image = trans_F.rotate(resized_image, rotate_angle * 90)
                    resized_density = trans_F.rotate(resized_density, rotate_angle * 90)
                for i in range(rotate_angle):
                    new_x = dots_new[:, 1]
                    new_y = self.output_w - dots_new[:, 0]
                    dots_new = torch.stack((new_x, new_y))
                    dots_new = dots_new.permute((1, 0))

            if "colour_jitter" in self.trans:
                jitter = T.ColorJitter(brightness=0.0, hue=0.25)
                resized_image = jitter(resized_image)
            if "rgb_scramble" in self.trans:
                chan_indexs = [0, 1, 2]
                random.shuffle(chan_indexs)
                resized_image = resized_image[chan_indexs, :]

        if self.split_up_img != 1:
            split_ims = []
            split_cnts = []
            for i in range(self.split_up_img):
                for j in range(self.split_up_img):

                    dts = dots_new
                    llx = i * self.img_size[0]
                    ulx = (i + 1) * self.img_size[0]
                    lly = j * self.img_size[1]
                    uly = (j + 1) * self.img_size[1]

                    im_ij = resized_image[:, llx:ulx, lly:uly,].clone()

                    dts_x = dts[:, 0]
                    dts_y = dts[:, 1]

                    dts_x_inds = torch.where(dts_x > lly, 1, 0)
                    dts_x_inds = torch.where(dts_x < uly, dts_x_inds, 0)

                    dts_y_inds = torch.where(dts_y > llx, 1, 0)
                    dts_y_inds = torch.where(dts_y < ulx, dts_y_inds, 0)

                    dts_inds = dts_y_inds * dts_x_inds
                    dts_ij = dts[dts_inds == 1, :]
                    dts_ij[:, 0] -= lly
                    dts_ij[:, 1] -= llx
                    cnt_ij = dts_ij.shape[0]
                    split_ims.append(im_ij)
                    split_cnts.append(torch.tensor(cnt_ij))
            resized_image = torch.stack(split_ims)
            gt_cnt = torch.stack(split_cnts)
            count = gt_cnt
        else:
            count = dots_new.shape[0]

        sample = {
            "image": resized_image,
            "boxes": boxes,
            "gt_density": resized_density,
            "dots": dots_new,
            "count": count,
        }
        return sample
