import numpy as np
import torch
import torchvision.transforms as T

from utils.data_utils import denormalize


def get_layer_maps(i, img_size, intermediate_image, input):

    resize_im = T.Resize(img_size, T.InterpolationMode.NEAREST)
    intermediate_image = resize_im(intermediate_image)

    img_i = input[i].clone()
    img_i = denormalize(img_i)

    img_i -= torch.min(img_i)
    img_i /= torch.max(img_i)
    if intermediate_image.shape[1] <= 3 and intermediate_image.shape[1] > 1:
        zz = torch.zeros_like(intermediate_image[0, 0].squeeze())

        vis_density_overlap = torch.stack((zz, zz, zz))
        for j in range(intermediate_image.shape[1]):
            ii_i = intermediate_image[i, j].squeeze()
            ii_i -= torch.min(ii_i)
            if torch.max(ii_i) > 0.1:
                ii_i /= torch.max(ii_i)
                if j == 0:
                    vis_density_overlap[j] = ii_i
                elif j == 1:
                    vis_density_overlap[2] = ii_i
                elif j == 2:
                    vis_density_overlap[1] = ii_i

        vis_density_overlap /= 2
        vis_density_overlap += img_i / 2

        vis_density_overlap = (
            vis_density_overlap.permute(1, 2, 0).detach().cpu().numpy()
        )
        vis_density_overlap *= 255
        vis_density_overlap = vis_density_overlap.astype(np.uint8).copy()

    return vis_density_overlap
