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
        self.inception = torchvision.models.inception_v3(pretrained=True)

        self.inception.fc = torch.nn.Linear(self.inception.fc.in_features, len(dataset.classes))
        self.inception.aux_logits = False
        self.inception.AuxLogits = None
        num_ftrs = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(num_ftrs, 300)
        # Handle the primary net
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, 300)

        # we need the feature map of the last conv layer,
        # so we remove the last two layers of resnet (average pool and fc)
        # modules = list(inception.children())[:-1]
        # self.inception = nn.Sequential(*modules)

        # resize input images with different size to fixed size
        # r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.
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
        return feature_map[0]

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
        feature_map = feature_map.permute(0, 2, 3,
                                          1)  # (batch_size, encoded_image_size = 7, encoded_image_size = 7, 2048)

        batch_size = feature_map.size(0)
        encoder_dim = feature_map.size(-1)
        num_pixels = feature_map.size(1) * feature_map.size(2)  # encoded_image_size * encoded_image_size = 49

        # flatten image
        feature_map = feature_map.view(batch_size, num_pixels,
                                       encoder_dim)  # (batch_size, num_pixels = 49, encoder_dim = 2048)

        return feature_map


class EncoderInceptionV3(nn.Module):
    """
    Implementation of the encoder proposed in paper [1].

    Parameters
    ----------
    encoded_image_size : int
        Size of resized feature map

    embed_dim : int
        Dimention of the output feature (same as dimension of word embeddings)

    References
    ----------
    1. "`Show and Tell: A Neural Image Caption Generator. \
        <https://arxiv.org/abs/1411.4555>`_" Oriol Vinyals, et al. CVPR 2015.
    """

    def __init__(self, encoded_image_size: int = 7, embed_dim: int = 512) -> None:
        super(EncoderInceptionV3, self).__init__()
        self.CNN = Inceptionv3(encoded_image_size)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=encoded_image_size,
            stride=encoded_image_size
        )
        self.output_layer = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim, momentum=0.01)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor (batch_size, 3, image_size=256, image_size=256)
            Input image

        Returns
        -------
        out : torch.Tensor (batch_size, embed_dim=512)
            Feature of this image
        """
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        batch_size = feature_map.size(0)
        out = self.avg_pool(feature_map).view(batch_size, -1)  # (batch_size, 2048)
        out = self.output_layer(out)  # (batch_size, embed_dim = 512)
        return out
