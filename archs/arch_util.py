# Nội dung cho file archs/arch_util.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Options for network modules
net_opt = {
    "norm": "in",  # in, bn, ln
    "act": "gelu",  # relu, lrelu, prelu, gelu, silu
    "v_act": "gelu",  # relu, lrelu, prelu, gelu, silu
    "w_act": "gelu",  # relu, lrelu, prelu, gelu, silu
    "bias": True,
    "c_bias": True,
    "conv_bias": True,
    "train_size": [1, 3, 256, 256],
}

class DySample(nn.Module):
    """
    Official implementation of Dynamic Upsampling
    """

    def __init__(
        self,
        in_ch: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        act: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        if style == "lp":
            self.offset = nn.Conv2d(in_ch, 2 * groups, 1, 1, 0, bias=True)
            self.mask = nn.Conv2d(in_ch, groups, 1, 1, 0, bias=True)
        elif style == "mask":
            self.mask = nn.Conv2d(in_ch, scale * scale, 1, 1, 0, bias=True)
        else:
            self.weight = nn.Conv2d(in_ch, groups, 1, 1, 0, bias=True)

        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "lp":
            # generate offset and mask
            offset = self.offset(x)
            mask = torch.sigmoid(self.mask(x))
            # deconvolution
            out = self.deform_conv(x, offset, mask)
        elif self.style == "mask":
            mask = self.act(self.mask(x))
            out = F.pixel_shuffle(x * mask, self.scale)
        else:
            weight = self.act(self.weight(x))
            out = F.pixel_shuffle(x * weight, self.scale)

        return out

    def deform_conv(
        self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        deformable convolution
        """
        b, c, h, w = x.shape
        g = self.groups
        # pre-sample features
        x_ = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        # normal convolution
        # b, c*s*s, h, w
        feat = (
            x_.reshape(b, c, self.scale, h, self.scale, w)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, self.scale * self.scale, c, h * w)
        )
        # b, 2*g, h, w -> b, g, 2, h, w -> b, g, h, w, 2
        offset = offset.reshape(b, g, 2, h, w).permute(0, 1, 3, 4, 2)
        # b, g, h, w
        mask = mask.reshape(b, g, h, w)
        # b, g, h, w, 2
        coords = self.create_grid(h, w, device=x.device).expand(b, g, -1, -1, -1)
        # b, g, h, w, 2
        coords = coords + offset
        # normalize
        coords = (coords / torch.tensor([w, h], device=x.device) - 0.5) * 2
        # b, g, c, h*w
        feat = feat.reshape(b, self.scale, self.scale, c, h * w).permute(0, 1, 2, 3, 4)
        # b, g, c, h*w
        feat = self.group(feat, g)
        # b, g, h*w, c
        feat = feat.permute(0, 1, 3, 2)
        # bilinear interpolate
        # b, g, h*w, c
        out = F.grid_sample(
            feat.reshape(b * g, h * w, 1, c),
            coords.reshape(b * g, h, w, 2).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, g, h * w, c)
        # b, g, h, w, c
        out = out.reshape(b, g, h, w, c) * mask.unsqueeze(-1)
        # b, c, h, w
        out = self.unsplit(out, c)

        return out

    def create_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device), torch.arange(0, w, device=device)
        )
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        return grid

    def group(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        b, s1, s2, c, hw = x.shape
        return x.reshape(b, s1 * s2, c, hw)[:, :groups, :, :]

    def unsplit(self, x: torch.Tensor, c: int) -> torch.Tensor:
        b, g, h, w, _ = x.shape
        return x.reshape(b, g, h, w, -1).sum(1).permute(0, 3, 1, 2)
