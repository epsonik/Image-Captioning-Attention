from torch import nn

from config import config
from .resnet import EncoderResNet, AttentionEncoderResNet, AdaptiveAttentionEncoderResNet
from .densenet import AttentionEncoderDenseNet201, EncoderDenseNet201


def make(embed_dim: int) -> nn.Module:
    """
    Make an encoder

    Parameters
    ----------
    embed_dim : int
        Dimention of word embeddings
    """
    model_name = config.caption_model
    pretrained_encoder = config.pretrained_encoder

    if model_name == 'show_tell':
        model = EncoderResNet(embed_dim=embed_dim)
        if pretrained_encoder == 'DenseNet201':
            model = EncoderDenseNet201()
    elif model_name == 'att2all':
        model = AttentionEncoderResNet()
        if pretrained_encoder == 'DenseNet201':
            model = AttentionEncoderDenseNet201()

    elif model_name == 'adaptive_att' or model_name == 'spatial_att':
        model = AdaptiveAttentionEncoderResNet(
            decoder_dim=config.decoder_dim,
            embed_dim=embed_dim
        )
    else:
        raise Exception("Model not supported: ", model_name)

    return model
