from torch import nn

from config import config
from .inception import EncoderInceptionV3, AttentionEncoderInceptionV3
from .resnet import EncoderResNet, AttentionEncoderResNet, AdaptiveAttentionEncoderResNet
from .resnet152 import EncoderResNet152, AttentionEncoderResNet152, AdaptiveAttentionEncoderResNet152
from .densenet import AttentionEncoderDenseNet201, EncoderDenseNet201, AdaptiveAttentionEncoderDenseNet201
from .densenet121 import AttentionEncoderDenseNet121, EncoderDenseNet121, AdaptiveAttentionEncoderDenseNet121
from .densenet161 import AttentionEncoderDenseNet161, EncoderDenseNet161, AdaptiveAttentionEncoderDenseNet161
from .regnet import AttentionEncoderRegnet32, EncoderRegnet32, AdaptiveAttentionEncoderRegNet32
from .regnet16 import AttentionEncoderRegnet16, EncoderRegnet16, AdaptiveAttentionEncoderRegnet16


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
            model = EncoderDenseNet201(embed_dim=embed_dim)
        if pretrained_encoder == 'DenseNet121':
            model = EncoderDenseNet121(embed_dim=embed_dim)
        if pretrained_encoder == 'DenseNet161':
            model = EncoderDenseNet161(embed_dim=embed_dim)
        if pretrained_encoder == 'Resnet152':
            model = EncoderResNet152(embed_dim=embed_dim)
        if pretrained_encoder == 'InceptionV3':
            model = EncoderInceptionV3(embed_dim=embed_dim)
    elif model_name == 'att2all':
        model = AttentionEncoderResNet()
        if pretrained_encoder == 'Resnet152':
            model = AttentionEncoderResNet152()
        if pretrained_encoder == 'DenseNet201':
            model = AttentionEncoderDenseNet201()
        if pretrained_encoder == 'DenseNet121':
            model = AttentionEncoderDenseNet121()
        if pretrained_encoder == 'DenseNet161':
            model = AttentionEncoderDenseNet161()
        if pretrained_encoder == 'Regnet32':
            model = AttentionEncoderRegnet32()
        if pretrained_encoder == 'Regnet16':
            model = AttentionEncoderRegnet16()
    elif model_name == 'adaptive_att' or model_name == 'spatial_att':
        model = AdaptiveAttentionEncoderResNet(
            decoder_dim=config.decoder_dim,
            embed_dim=embed_dim
        )
        if pretrained_encoder == 'Resnet101':
            model = AdaptiveAttentionEncoderResNet(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )
        if pretrained_encoder == 'Resnet152':
            model = AdaptiveAttentionEncoderResNet152(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )
        if pretrained_encoder == 'DenseNet201':
            model = AdaptiveAttentionEncoderDenseNet201(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )
        if pretrained_encoder == 'DenseNet121':
            model = AdaptiveAttentionEncoderDenseNet121(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )
        if pretrained_encoder == 'DenseNet161':
            model = AdaptiveAttentionEncoderDenseNet161(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )
        if pretrained_encoder == 'Regnet32':
            model = AdaptiveAttentionEncoderRegNet32(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )
        if pretrained_encoder == 'Regnet16':
            model = AdaptiveAttentionEncoderRegnet16(
                decoder_dim=config.decoder_dim,
                embed_dim=embed_dim
            )


    else:
        raise Exception("Model not supported: ", model_name)

    return model
