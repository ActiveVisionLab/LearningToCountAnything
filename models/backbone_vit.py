import math
import torch
import torch.nn as nn
import timm
import math


class ViTExtractor(nn.Module):
    def __init__(self, config):
        """
        model_name: [str] base_model-[facet]_[layer]_[bin]_[stride]
            base_model: (vit_small_patch16_224_dino | vit_small_patch8_224_dino | vit_base_patch16_224_dino | vit_base_patch8_224_dino
                        vit_base_patch8_224 | vit_base_patch16_224)
            facet: (key | query | value | attn | token)
            layer: [int] 0-11, NOTE if facet is attn, the layer is automatically set to 11 no matter what is defined in config
            bin: [int] 0 or 1
            stride: [int] any stride that is divisible by image size
        """

        # vit_small_patch8_224_dino-attn_
        super(ViTExtractor, self).__init__()
        self.base_model = config["base_model"]
        self.facet = config["facet"]
        self.layer = config["layer"]
        self.bin = config["bin"]
        self.stride = config["stride"]

        base_model_ = timm.create_model(self.base_model, pretrained=True)
        self.create_from_base_model(
            base_model_, facet=self.facet, layer=self.layer, stride=self.stride
        )
        del base_model_

        self.p = self.patch_embed.patch_size
        self.stride = self.patch_embed.proj.stride

        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    def create_from_base_model(self, base_model, facet, layer, stride) -> None:

        # copy various modules from base model
        self.num_features = base_model.embed_dim
        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        # self.dist_token = base_model.dist_token
        self.pos_drop = base_model.pos_drop
        self.pos_embed = base_model.pos_embed
        # self.norm = base_model.norm  # this probably does nothing

        # cut block module
        self.blocks = ViTExtractor.fix_blocks(base_model.blocks, facet, layer)

        patch_size = self.patch_embed.patch_size[0]
        h, w = self.patch_embed.img_size

        if stride == patch_size:
            pass
        else:
            # fix patch embedding
            self.patch_embed.proj.stride = (stride, stride)
            # fix positional encoding
            self.pos_embed = nn.Parameter(
                data=ViTExtractor.fix_pos_embed(
                    base_model.pos_embed, patch_size, stride, h, w
                )
            )

    @staticmethod
    def fix_blocks(blocks, facet, layer):
        out = []
        for i in range(layer):
            out.append(blocks[i])
        if facet == "token":
            out.append(blocks[layer])
        elif facet == "query" or facet == "key" or facet == "value" or facet == "attn":
            out.append(ViTExtractor.partial_block(blocks[layer], facet))
        return nn.Sequential(*out)

    @staticmethod
    def partial_block(block, facet):
        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        elif facet == "attn":
            facet_idx = None
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        class PartialBlock(nn.Module):
            def __init__(self):
                super(PartialBlock, self).__init__()
                self.norm1 = block.norm1
                self.attn = block.attn

            def forward(self, x):
                x = self.norm1(x)
                B, N, C = x.shape
                qkv = (
                    self.attn.qkv(x)
                    .reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                if facet == "query" or facet == "key" or facet == "value":
                    return qkv[facet_idx]
                elif facet == "attn":
                    q, k, v = qkv.unbind(
                        0
                    )  # make torchscript happy (cannot use tensor as tuple)
                    attn = (q @ k.transpose(-2, -1)) * self.attn.scale
                    attn = attn.softmax(dim=-1)
                    attn = self.attn.attn_drop(attn)
                    return attn
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

        return PartialBlock()

    @staticmethod
    def fix_pos_embed(pos_embed, patch_size, stride, h, w):

        N = pos_embed.shape[1] - 1
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]

        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride
        h0 = 1 + (h - patch_size) // stride

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _extract_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        # if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
        #    x = torch.cat(
        #        (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1
        #    )
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)

        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p[0]) // self.stride[0],
            1 + (W - self.p[1]) // self.stride[1],
        )
        return x

    def _log_bin(self, x, hierarchy=2, mode="long"):
        """
        Create a log binning feature map
        x [torch.Tensor] (B x he x )
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(
            B, bin_x.shape[1], self.num_patches[0], self.num_patches[1]
        )
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(
                win_size, stride=1, padding=win_size // 2, count_include_pad=False
            )
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros(
            (B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])
        ).to(x.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(
                            x - kernel_size, x + kernel_size + 1, kernel_size
                        ):
                            if i == y and j == x and k != 0:
                                continue
                            if (
                                0 <= i < self.num_patches[0]
                                and 0 <= j < self.num_patches[1]
                            ):
                                bin_x[
                                    :,
                                    part_idx
                                    * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[
                                    :,
                                    part_idx
                                    * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        bin_x = (
            bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        )
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(
        self, batch, layer=11, facet="key", bin=False, include_cls=False
    ):
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """

        x = self._extract_features(batch)
        if facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert (
                not bin
            ), "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = (
                x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
            )  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, img, reduction="mean", heads=[0, 2, 4, 5]):
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param img: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert (
            self.base_model == "vit_small_patch8_224_dino"
        ), "According to DINO paper, the only model support attn map is ViT-S/8"
        feat = self._extract_features(img)  # Bxhxtxt
        cls_attn_map = feat[:, heads, 0, 1:]  # Bxhx(t-1)
        if reduction == "mean":
            cls_attn_map = cls_attn_map.mean(dim=1)  # Bx(t-1)
            temp_mins, temp_maxs = (
                cls_attn_map.min(dim=1, keepdim=True)[0],
                cls_attn_map.max(dim=1, keepdim=True)[0],
            )
            cls_attn_maps = (cls_attn_map - temp_mins) / (
                temp_maxs - temp_mins
            )  # normalize to range [0,1]
            cls_attn_maps = cls_attn_maps.reshape(
                cls_attn_maps.shape[0], self.num_patches[0], self.num_patches[1]
            )
        elif reduction is None:
            temp_mins, temp_maxs = (
                cls_attn_map.min(dim=2, keepdim=True)[0],
                cls_attn_map.max(dim=2, keepdim=True)[0],
            )
            cls_attn_maps = (cls_attn_map - temp_mins) / (
                temp_maxs - temp_mins
            )  # normalize to range [0,1]
            cls_attn_maps = cls_attn_maps.reshape(
                cls_attn_maps.shape[0],
                cls_attn_maps.shape[1],
                self.num_patches[0],
                self.num_patches[1],
            )
        return cls_attn_maps

    def forward(self, x):
        if self.facet != "attn":
            feat = self.extract_descriptors(
                x, layer=self.layer, facet=self.facet, bin=self.bin
            )
            feat = feat.squeeze(1).permute(0, 2, 1)
            B, C, npatch = feat.shape
            N = int(math.sqrt(npatch))
            feat = feat.reshape(B, C, N, N)
        else:
            feat = self.extract_saliency_maps(x, reduction=None, heads=range(6))

        return feat
