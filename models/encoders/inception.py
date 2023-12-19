"""
Implementation of some encoders using ResNet-101 as backbone.
"""

import torch
from torch import nn
import torchvision
from typing import Tuple

class Inceptionv3(nn.Module):
    """
    (Pretrained) ResNet-101 network

    Parameters
    ----------
    encoded_image_size : int
        Size of the resized feature map
    """
    def __init__(self, encoded_image_size: int = 7):
        super(Inceptionv3, self).__init__()
        self.enc_image_size = encoded_image_size  # size of resized feature map

        # pretrained ResNet-101 model (on ImageNet)
        inception = torchvision.models.inception_v3(pretrained = True)

        # we need the feature map of the last conv layer,
        # so we remove the last two layers of resnet (average pool and fc)
        modules = list(inception.children())[:-3]
        self.inception = nn.Sequential(*modules)

        # resize input images with different size to fixed size
        # r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor (batch_size, 3, image_size=256, image_size=256)
            Input image

        Returns
        -------
        feature_map : torch.Tensor (batch_size, 2048, encoded_image_size=7, encoded_image_size=7)
            Feature map after resized
        """
        feature_map = self.inception(images)  # (batch_size, 2048, image_size/32, image_size/32)
        feature_map = self.adaptive_pool(feature_map)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        return feature_map

    def fine_tune(self, fine_tune: bool = True) -> None:
        """
        Parameters
        ----------
        fine_tune : bool
            Fine-tune CNN (conv block 2-4) or not
        """
        for param in self.inception.parameters():
            param.requires_grad = False
        # only fine-tune conv block 2-4
        for module in list(self.inception.children())[5:]:
            for param in module.parameters():
                param.requires_grad = fine_tune


class AttentionEncoderInceptionV3(nn.Module):
    """
    Implementation of the encoder proposed in paper [1]

    Parameters
    ----------
    encoded_image_size : int
        Size of resized feature map

    References
    ----------
    1. "`Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. \
        <https://arxiv.org/abs/1502.03044>`_" Kelvin Xu, et al. ICML 2015.
    """
    def __init__(self, encoded_image_size: int = 7) -> None:
        super(AttentionEncoderInceptionV3, self).__init__()
        self.CNN = Inceptionv3(encoded_image_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor (batch_size, 3, image_size=256, image_size=256)
            Input image

        Returns
        -------
        feature_map : torch.Tensor (batch_size, num_pixels=49, encoder_dim=2048)
            Feature map of the image
        """
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        feature_map = feature_map.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size = 7, encoded_image_size = 7, 2048)

        batch_size = feature_map.size(0)
        encoder_dim = feature_map.size(-1)
        num_pixels = feature_map.size(1) * feature_map.size(2)  # encoded_image_size * encoded_image_size = 49

        # flatten image
        feature_map = feature_map.view(batch_size, num_pixels, encoder_dim)  # (batch_size, num_pixels = 49, encoder_dim = 2048)

        return feature_map


