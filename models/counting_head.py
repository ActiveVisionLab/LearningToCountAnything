import torch
import torch.nn as nn
from einops import rearrange


class CountingHeadRegression(nn.Module):
    def __init__(self, feature_dim, resloution=28, c=3, complexity="simple"):
        super().__init__()

        self.ch = c
        self.complexity = complexity
        if self.complexity == "simple":
            self.conv1 = nn.Conv2d(
                feature_dim, feature_dim, 3, 1, 1, padding_mode="reflect"
            )
            self.l_reg = nn.Linear(feature_dim, self.ch)
            self.l_reg2 = nn.Linear(self.ch * resloution * resloution, 1)

        if self.complexity == "complex":
            self.bn1 = nn.BatchNorm2d(feature_dim)
            self.bn2 = nn.BatchNorm2d(int(feature_dim / 2))
            self.bn3 = nn.BatchNorm2d(int(feature_dim / 4))
            self.bn4 = nn.BatchNorm2d(int(feature_dim / 8))

            self.conv1 = nn.Conv2d(
                feature_dim, feature_dim, 3, 1, 1, padding_mode="reflect"
            )
            self.conv2 = nn.Conv2d(
                feature_dim, int(feature_dim / 2), 3, 1, 1, padding_mode="reflect"
            )
            self.conv3 = nn.Conv2d(
                int(feature_dim / 2),
                int(feature_dim / 4),
                3,
                1,
                1,
                padding_mode="reflect",
            )
            self.conv4 = nn.Conv2d(
                int(feature_dim / 4),
                int(feature_dim / 8),
                3,
                1,
                1,
                padding_mode="reflect",
            )
            self.l_reg = nn.Linear(int(feature_dim / 8), self.ch)
            self.l_reg2 = nn.Linear(self.ch * resloution * resloution, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        latent_head_feat = None

        if self.complexity == "simple":
            x = torch.sigmoid(self.conv1(x))
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.l_reg(x)
            x = torch.relu(x)
            x = rearrange(x, "b (h w) c -> b (c h w)", b=b, h=h, w=w, c=self.ch)
            latent_head_feat = x.clone()
            x = self.l_reg2(x)

        if self.complexity == "complex":
            x1 = torch.relu(self.bn1(self.conv1(x)))
            x2 = torch.relu(self.bn2(self.conv2(x1)))
            x3 = torch.relu(self.bn3(self.conv3(x2)))
            x4 = torch.relu(self.bn4(self.conv4(x3)))
            x = rearrange(x4, "b c h w -> b (h w) c")
            x = self.l_reg(x)
            x = torch.relu(x)
            x = rearrange(x, "b (h w) c -> b (c h w)", b=b, h=h, w=w, c=self.ch)
            latent_head_feat = x.clone()
            x = self.l_reg2(x)
            x = x.squeeze(1)

        return x, latent_head_feat


class CountingHeadLinearProbe(nn.Module):
    def __init__(self, feature_dim, resloution=28, n_outputs_gt_select=1):
        super().__init__()

        if n_outputs_gt_select != 1:
            self.linear = nn.Linear(
                feature_dim * resloution * resloution, n_outputs_gt_select
            )
        else:
            self.linear = nn.Linear(feature_dim * resloution * resloution, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.linear(x)
        x = x.squeeze()  # (b, 1) -> b
        return x, None
