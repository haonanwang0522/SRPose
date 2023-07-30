import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from mmpose.models.builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class BottomUpSimpleSRHead(nn.Module):
    """Bottom-up simple head.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        with_ae_loss (list[bool]): Option to use ae loss or not.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 tag_per_joint=True,
                 with_ae_loss=None,
                 extra=None,
                 loss_keypoint=None,
                 per_kp_emb=4,
                 pixel_shuffle=1):
        super().__init__()
        dim_tag = num_joints if tag_per_joint else 1
        self.pixel_shuffle = pixel_shuffle

        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        self.kp_encoder = nn.Sequential(build_conv_layer(
                            cfg=dict(type='Conv2d'),
                            in_channels=num_deconv_filters[-1] if num_deconv_layers>0 else in_channels,
                            out_channels=num_joints*per_kp_emb,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                            nn.ReLU())
        self.heatmap_head = build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=num_joints*per_kp_emb,
                    out_channels=num_joints*(pixel_shuffle**2),
                    kernel_size=9,
                    stride=1,
                    padding=4,
                    groups=num_joints)
        if with_ae_loss[0]:
            self.tag_head = build_conv_layer(
                        cfg=dict(type='Conv2d'),
                        in_channels=num_deconv_filters[-1] if num_deconv_layers>0 else in_channels,
                        out_channels=dim_tag,
                        kernel_size=1,
                        stride=1,
                        padding=0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def get_loss(self, outputs, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (list(torch.Tensor[N,K,H,W])): Multi-scale output heatmaps.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints(List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
        """

        losses = dict()

        heatmaps_losses, push_losses, pull_losses = self.loss(
            outputs, targets, masks, joints)

        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                if 'push_loss' not in losses:
                    losses['push_loss'] = push_loss
                else:
                    losses['push_loss'] += push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                if 'pull_loss' not in losses:
                    losses['pull_loss'] = pull_loss
                else:
                    losses['pull_loss'] += pull_loss

        return losses

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        final_outputs = []
        x = self.deconv_layers(x)
        tag = self.tag_head(x)
        x = self.kp_encoder(x)
        hp = self.heatmap_head(x)
        hp = F.pixel_shuffle(hp, self.pixel_shuffle)
        tag = F.interpolate(tag, hp.shape[2:])
        y = torch.cat([hp,tag], 1)
        final_outputs.append(y)
        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
