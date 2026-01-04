import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import SSIM
from copy import deepcopy
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        haar_weights = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[-0.5, -0.5], [0.5, 0.5]]],
            [[[-0.5, 0.5], [-0.5, 0.5]]],
            [[[0.5, -0.5], [-0.5, 0.5]]],
        ], dtype=torch.float32)
        self.register_buffer('weight', haar_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, stride=2)


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY and pred.shape[1] == 3:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred, target = pred / 255., target / 255.

        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return 40 + self.loss_weight * self.scale * torch.log(mse + 1e-8).mean()


class SCIMLoss(nn.Module):
    def __init__(self, patch_size=30, default_loss_weight=15.0, ssim_window_size=11,
                 data_range=1.0, sampling_mode='random', num_patches=1, **kwargs):
        super(SCIMLoss, self).__init__()
        self.patch_h = self.patch_w = patch_size
        self.default_loss_weight = default_loss_weight
        self.sampling_mode = sampling_mode
        self.num_patches = num_patches

        self.ssim_module = SSIM(
            data_range=data_range,
            size_average=True,
            win_size=ssim_window_size,
            nonnegative_ssim=True,
            channel=1
        )

    def forward(self, src_vec, tar_vec, loss_weight=None, num_patches=None):
        current_loss_weight = self.default_loss_weight if loss_weight is None else loss_weight
        b, c, h, w = src_vec.shape
        assert c == 1, "SCIMLoss only supports single-channel grayscale images."

        if h < self.patch_h or w < self.patch_w:
            return current_loss_weight * (1.0 - self.ssim_module(src_vec, tar_vec))

        num_patches_val = num_patches if num_patches is not None else self.num_patches

        rand_h = torch.randint(0, h - self.patch_h + 1, (b * num_patches_val,), device=src_vec.device)
        rand_w = torch.randint(0, w - self.patch_w + 1, (b * num_patches_val,), device=src_vec.device)

        batch_indices = torch.repeat_interleave(torch.arange(b, device=src_vec.device), num_patches_val)

        src_patches = []
        tar_patches = []
        for i in range(b * num_patches_val):
            h_start, w_start = rand_h[i], rand_w[i]
            h_end, w_end = h_start + self.patch_h, w_start + self.patch_w

            src_patches.append(src_vec[batch_indices[i]].unsqueeze(0)[:, :, h_start:h_end, w_start:w_end])
            tar_patches.append(tar_vec[batch_indices[i]].unsqueeze(0)[:, :, h_start:h_end, w_start:w_end])

        src_patches_batch = torch.cat(src_patches, dim=0)
        tar_patches_batch = torch.cat(tar_patches, dim=0)

        ssim_score = self.ssim_module(src_patches_batch, tar_patches_batch)
        loss = 1.0 - ssim_score
        return current_loss_weight * loss


class WaveletLoss(nn.Module):
    def __init__(self, loss_configs: dict):
        super(WaveletLoss, self).__init__()

        self.dwt = DWT()
        self.sub_losses = nn.ModuleDict()
        self.sub_weights = {}

        for band_name, config in loss_configs.items():
            current_config = deepcopy(config)

            loss_type_str = current_config.pop('type')
            weight = current_config.pop('weight')
            self.sub_weights[band_name] = weight

            loss_cls = globals()[loss_type_str]
            loss_instance = loss_cls(**current_config)
            self.sub_losses[band_name] = loss_instance

        for band_name, loss_module in self.sub_losses.items():
            loss_name = loss_module.__class__.__name__
            weight = self.sub_weights[band_name]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dwt = self.dwt(pred)
        target_dwt = self.dwt(target)

        sub_bands_pred = list(torch.split(pred_dwt, 1, dim=1))
        sub_bands_target = list(torch.split(target_dwt, 1, dim=1))

        total_loss = 0.0
        band_names = ['ll', 'lh', 'hl', 'hh']

        for i, band_name in enumerate(band_names):
            pred_band = sub_bands_pred[i]
            target_band = sub_bands_target[i]

            loss_fn = self.sub_losses[band_name]
            weight = self.sub_weights[band_name]

            total_loss += weight * loss_fn(pred_band, target_band)

        return total_loss