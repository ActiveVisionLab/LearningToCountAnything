import json
import logging
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as trans_F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision import transforms

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


class resizeImageBBoxesDensity(object):
    def __init__(self, CFG, train):
        self.output_h = CFG["img_size"][0]
        self.output_w = CFG["img_size"][1]
        self.img_mode = CFG["img_mode"]
        self.scale_and_crop_random = CFG["scale_and_crop_random"]
        self.trans = CFG["image_transforms"]
        if "increase_density" in self.trans:
            self.increase_density_ratio = CFG["increase_density_ratio"]
            self.increase_density_rotate = CFG["increase_density_rotate"]
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

            if not isinstance(density, torch.Tensor):
                density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
            resized_density = transforms.Resize(
                (resized_image.size[1], resized_image.size[0])
            )(density)
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

        elif self.img_mode == "scale_and_crop":
            if scale_factor_w < scale_factor_h:
                # needs to be squished more horizontaly than vertically
                # squish vertically then crop out horizontally

                resized_image = transforms.Resize(self.output_h)(image)
                if not isinstance(density, torch.Tensor):
                    density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
                resized_density = transforms.Resize(
                    (resized_image.size[1], resized_image.size[0])
                )(density)

                orig_count = torch.sum(density)
                new_count = torch.sum(resized_density)
                if new_count > 0:
                    resized_density = resized_density * (orig_count / new_count)

                if self.scale_and_crop_random:
                    crop_x = random.randint(0, resized_image.size[0] - self.output_w)
                else:
                    crop_x = int((resized_image.size[0] - self.output_w) / 2)

                resized_image = resized_image.crop(
                    (crop_x, 0, crop_x + self.output_w, self.output_h)
                )

                resized_density = resized_density[
                    :, :, :, crop_x : crop_x + self.output_w
                ]
                boxes = list()

                for i, box in enumerate(lines_boxes):
                    # box2 = [int(k*scale_factor) for k in box]
                    # if completely off screen delete
                    y1, x1, y2, x2 = (
                        box[0] * scale_factor_h,
                        box[1] * scale_factor_w,
                        box[2] * scale_factor_h,
                        box[3] * scale_factor_w,
                    )
                    if x1 < crop_x + self.output_w and x2 > crop_x:
                        x1 = np.clip(x1, crop_x, crop_x + self.output_w)
                        x2 = np.clip(x2, crop_x, crop_x + self.output_w)
                        boxes.append([0, y1, x1, y2, x2])
                    else:
                        boxes.append([0, 0, 0, 0, 0])

                boxes = torch.Tensor(boxes).unsqueeze(0)

                dots_new = [[], []]
                for dot in enumerate(dots):
                    dot_0 = dot[1][0] * scale_factor_w
                    dot_1 = dot[1][1] * scale_factor_h
                    if dot_0 > crop_x and dot_0 < crop_x + self.output_w:
                        dots_new[0].append(dot_0)
                        dots_new[1].append(dot_1)
                    # else:
                    #     print("", dot_0, dot_1, crop_x, crop_x + self.output_w)
                dots_new = torch.tensor(dots_new).permute(1, 0)

            elif scale_factor_w > scale_factor_h:
                print("!!!need to be implemented")
                exit()
        elif self.img_mode == "pad0":
            if scale_factor_w < scale_factor_h:
                # needs to be squished more horizontaly than vertically
                # squish vertically then crop out horizontally

                resized_image = transforms.Resize(
                    [int(H * scale_factor_w), self.output_w]
                )(image)
                if not isinstance(density, torch.Tensor):
                    density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
                resized_density = transforms.Resize(
                    (resized_image.size[1], resized_image.size[0])
                )(density)

                orig_count = torch.sum(density)
                new_count = torch.sum(resized_density)
                if new_count > 0:
                    resized_density = resized_density * (orig_count / new_count)

                pad_amount = self.output_h - int(H * scale_factor_w)
                pad_0 = int(pad_amount / 2)
                pad_1 = pad_amount - int(pad_amount / 2)

                padding_0 = torch.zeros([1, 1, pad_0, self.output_w])
                padding_1 = torch.zeros([1, 1, pad_1, self.output_w])

                resized_density = torch.cat(
                    (padding_0, resized_density, padding_1), dim=2
                )

                resized_image = ImageOps.pad(
                    resized_image,
                    (self.output_w, self.output_h),
                    color=None,
                    centering=(0.5, 0.5),
                )

                #
                boxes = list()

                for i, box in enumerate(lines_boxes):
                    y1, x1, y2, x2 = (
                        box[0] * scale_factor_w + pad_0,
                        box[1] * scale_factor_w,
                        box[2] * scale_factor_w + pad_0,
                        box[3] * scale_factor_w,
                    )
                    boxes.append([0, y1, x1, y2, x2])

                boxes = torch.Tensor(boxes).unsqueeze(0)

                dots_new = [[], []]
                for dot in enumerate(dots):
                    dot_0 = dot[1][0] * scale_factor_w
                    dot_1 = dot[1][1] * scale_factor_w + pad_0
                    dots_new[0].append(dot_0)
                    dots_new[1].append(dot_1)
                dots_new = torch.tensor(dots_new).permute(1, 0)

            elif scale_factor_w > scale_factor_h:
                print("This case shouldnt occur")
                exit()

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
                    if self.increase_density_rotate:
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
                rotate_angle = int(random.random() * 4)
                if rotate_angle != 0:
                    resized_image = trans_F.rotate(resized_image, rotate_angle * 90)
                    resized_density = trans_F.rotate(resized_density, rotate_angle * 90)

            if "colour_jitter" in self.trans:
                jitter = T.ColorJitter(brightness=0.0, hue=0.25)
                resized_image = jitter(resized_image)
            if "rgb_scramble" in self.trans:
                chan_indexs = [0, 1, 2]
                random.shuffle(chan_indexs)
                resized_image = resized_image[chan_indexs, :]
        sample = {
            "image": resized_image,
            "boxes": boxes,
            "gt_density": resized_density,
            "dots": dots_new,
        }
        return sample


class CountingDatasetFSC147(Dataset):
    def __init__(self, CFG, train, transform):
        self.CFG = CFG
        self.img_size = CFG["img_size"]
        self.train = train
        self.img_channels = CFG["img_channels"]
        self.dataset = CFG["dataset"]

        data_path = CFG["data_path"] + "FSC-147/"
        anno_file = data_path + "annotation_FSC147_384.json"
        data_split_file = data_path + "Train_Test_Val_FSC_147.json"
        self.im_dir = data_path + "images_384_VarV2"
        self.gt_dir = data_path + "gt_density_map_adaptive_384_VarV2"

        self.annotations = None
        data_split = None

        # get annotations
        if anno_file != None:
            with open(anno_file) as f:
                self.annotations = json.load(f)

        # get datasplit
        if data_split_file != None:
            with open(data_split_file) as f:
                data_split = json.load(f)

        if data_split_file == None:
            # self.im_ids = os.listdir(self.im_dir)
            self.im_ids = [
                f
                for f in os.listdir(self.im_dir)
                if os.path.isfile(self.im_dir + "/" + f)
            ]

        else:
            if self.train:
                self.im_ids = data_split["train"]
            else:
                self.im_ids = data_split[CFG["test_split"]]

        if (
            CFG["exclude_imgs_with_counts_over_500"]
            or CFG["exclude_imgs_with_counts_over_1000"]
        ):
            self.remove_im_ids_if_high_density(CFG)

        # if we are excluding images where the average bbox size or aspect ratio are in the extremes
        if (
            CFG["bboxes_sizes_to_look_at"] != "all"
            or CFG["bboxes_aspect_ratio_to_look_at"] != "all"
        ):
            self.remove_im_ids_based_on_bbox_constraints(CFG)

        print(
            "{} set, size:{}, eg: {}".format(
                "train" if train else "test", len(self.im_ids), self.im_ids[:5]
            )
        )

        self.transform = transform

    def remove_im_ids_if_high_density(self, CFG):
        # remove image ids from list if there are over 500 or over 1000 instances
        over_500s, over_1000s = FSC147_over_500_over_1000()

        how_many_remove_1000 = 0
        for idx in over_1000s:
            if idx in self.im_ids:
                how_many_remove_1000 += 1
                self.im_ids.remove(idx)
        print(
            f"{how_many_remove_1000} images excluded for having a count of more than 1000",
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

                    # allows for quick visulaisation of the boundaries
                    # if CFG["bbox_big_area"] * 0.95 < mean_area:
                    # if CFG["bbox_small_area"] * 1.01 > mean_area:
                    #     image = Image.open("{}/{}".format(self.im_dir, id))
                    #     image.load()
                    #     plt.imshow(image)
                    #     plt.show()

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
            "{}/{} images remaining after removing size and/or aspect ratio outliers".format(
                len(selective_ids), len(self.im_ids)
            )
        )

        self.im_ids = selective_ids
        print("selective_ids", len(selective_ids))

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):

        im_id = self.im_ids[idx]
        image = Image.open("{}/{}".format(self.im_dir, im_id))
        image.load()

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
            gt_cnt = -1
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

        gt_cnt = sample["dots"].shape[0]

        if self.img_channels == 1:
            sample["image"] = torch.mean(sample["image"], dim=0).unsqueeze(0)

        img_id = torch.tensor(int(im_id.split(".jpg")[0]))
        return (
            sample["image"],
            sample["boxes"],
            sample["gt_density"],
            gt_cnt,
            img_id,
        )


class TestImagesDataset(Dataset):
    def __init__(self, CFG, train):
        # used to run small numbers of images from a seperate directory

        self.img_size = CFG["img_size"]
        self.train = train
        self.im_dir = "/data/test_ims"
        self.im_ids = [
            f for f in os.listdir(self.im_dir) if os.path.isfile(self.im_dir + "/" + f)
        ]

        print(
            "{} set, size:{}, eg: {}".format(
                "train" if train else "test", len(self.im_ids), self.im_ids[:5]
            )
        )
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
        image = Image.open("{}/{}".format(self.im_dir, im_id))
        image.load()
        image = self.trans(image)

        fl_name = im_id.split(".jpg")[0]
        img_id, gt_cnt = fl_name.split("_")
        gt_cnt = torch.Tensor([int(gt_cnt)])

        _ = torch.Tensor((1))
        return (
            image,
            _,
            _,
            gt_cnt,
            img_id,
        )


class CountingDatasetCOCO(Dataset):
    """"""

    def __init__(self, CFG, train, transform, fold):
        self.CFG = CFG
        self.img_size = CFG["img_size"]
        self.train = train
        self.img_channels = CFG["img_channels"]
        self.dataset = CFG["dataset"]

        self.data_path = CFG["data_path"]
        self.im_dir = self.data_path + "/coco/images/val2017"
        img_cats_count_file = self.data_path + "/coco/coco_id_cats_count.json"
        self.gt_dir = None

        with open(img_cats_count_file, "r") as f:
            img_id_instance_cats_detailed = json.load(f)

        classes_in_fold = []
        for c in range(100):
            if (c - 1) % 4 == fold and not self.train:
                classes_in_fold.append(c)
            elif (c - 1) % 4 != fold and self.train:
                classes_in_fold.append(c)

        img_id_instance_cats_detailed_fold = {}
        for img_id_and_cat in list(img_id_instance_cats_detailed.keys()):
            _, cat = img_id_and_cat.split("_")
            if int(cat) in classes_in_fold:

                if CFG["dataset"] == "coco_5":
                    if img_id_instance_cats_detailed[img_id_and_cat] >= 5:
                        img_id_instance_cats_detailed_fold[
                            img_id_and_cat
                        ] = img_id_instance_cats_detailed[img_id_and_cat]

                else:
                    img_id_instance_cats_detailed_fold[
                        img_id_and_cat
                    ] = img_id_instance_cats_detailed[img_id_and_cat]

        self.im_ids = list(img_id_instance_cats_detailed_fold.keys())
        self.img_id_instance_cats_detailed = img_id_instance_cats_detailed_fold
        self.trans = CFG["image_transforms"]
        self.transform = transform

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):

        im_id_and_cnt = self.im_ids[idx]
        gt_cnt = self.img_id_instance_cats_detailed[im_id_and_cnt]
        im_id, _im_cat = im_id_and_cnt.split("_")
        im_id = im_id.rjust(12, "0")

        image = Image.open("{}/{}.jpg".format(self.im_dir, im_id))
        image.load()
        image = transforms.ToTensor()(image)

        # in the rare case of a single channel coco image stack to make RGB
        if image.size()[0] == 1:
            image = torch.cat([image, image, image], dim=0)
        else:
            # normalise all colour images
            image = Normalize_tensor(image)

        resized_image = transforms.Resize((self.img_size[0], self.img_size[1]))(image)

        if self.train:
            if "reflect_rotate" in self.trans:
                if random.random() > 0.5:
                    resized_image = trans_F.hflip(resized_image)
                rotate_angle = int(random.random() * 4)
                if rotate_angle != 0:
                    resized_image = trans_F.rotate(resized_image, rotate_angle * 90)

            if "colour_jitter" in self.trans:
                jitter = T.ColorJitter(brightness=0.0, hue=0.25)
                resized_image = jitter(resized_image)
            if "rgb_scramble" in self.trans:
                chan_indexs = [0, 1, 2]
                random.shuffle(chan_indexs)
                resized_image = resized_image[chan_indexs, :]

        # make a blank gt_density not used for training
        gt_density = torch.zeros([1, 1, self.img_size[0], self.img_size[1]])
        img_id_tens = torch.tensor(int(im_id))

        return resized_image, torch.zeros([1]), gt_density, gt_cnt, img_id_tens


def get_loader_counting(CFG):

    TransformTest = transforms.Compose([resizeImageBBoxesDensity(CFG, train=False)])

    if CFG["dataset"] == "coco" or CFG["dataset"] == "coco_5":
        testset = CountingDatasetCOCO(
            CFG, train=False, transform=None, fold=CFG["coco_fold"]
        )
    elif CFG["dataset"] == "test_ims":
        testset = TestImagesDataset(CFG, train=False)
    else:
        testset = CountingDatasetFSC147(CFG, train=False, transform=TransformTest)

    test_sampler = SequentialSampler(testset)
    test_loader = (
        DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=CFG["eval_batch_size"],
            num_workers=4,
            pin_memory=True,
            drop_last=CFG["drop_last"],
        )
        if testset is not None
        else None
    )

    if not CFG["test"]:
        TransformTrain = transforms.Compose([resizeImageBBoxesDensity(CFG, train=True)])

        if CFG["dataset"] == "coco" or CFG["dataset"] == "coco_5":
            trainset = CountingDatasetCOCO(
                CFG, train=True, transform=None, fold=CFG["coco_fold"]
            )

        elif CFG["dataset"] == "test_ims":
            trainset = TestImagesDataset(CFG, train=True)
        else:
            trainset = CountingDatasetFSC147(CFG, train=True, transform=TransformTrain)

        train_sampler = RandomSampler(trainset)
        train_loader = DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=CFG["train_batch_size"],
            num_workers=4,
            pin_memory=True,
            drop_last=CFG["drop_last"],
        )

        return train_loader, test_loader
    return test_loader


def FSC147_over_500_over_1000():
    # return lists of images with over 500 and over 1000 count

    # all images in train, val and test set with 500<count<1000
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
    # all images in train, val and test set with 1000<count
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
