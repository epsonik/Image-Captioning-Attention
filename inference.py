import json
import numpy as np
from typing import Dict
# from scipy.misc import imread, imresize
from imageio import imread
from PIL import Image
import torch
import os

import torchvision.transforms as transforms

from utils import visualize_att_beta, visualize_att
from config import config

device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")

device = torch.device(
    "cuda:2" if torch.cuda.is_available() else "cpu")

data_f = os.path.join(config.base_path, "data")
# word map, ensure it's the same the data was encoded with and the model was trained with
wordmap_path = os.path.join(data_f, "evaluation", 'wordmap' + '.json')
# load word map (word2ix)
with open(wordmap_path, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word


def generate_caption(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: str,
    word_map: Dict[str, int],
    caption_model: str,
    beam_size: int = 3,
    pretrained_encoder: str = 'Resnet101'
):
    """
    Generate a caption on a given image using beam search.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder model

    decoder : torch.nn.Module
        Decoder model

    image_path : str
        Path to image

    word_map : Dict[str, int]
        Word map

    beam_size : int, optional, default=3
        Number of sequences to consider at each decode-step

    return:
        seq: caption
        alphas: weights for visualization
    """

    # read and process an image
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # img = imresize(img, (256, 256))
    resized_size = 256
    if pretrained_encoder == 'Regnet32' or pretrained_encoder == 'Regnet16':
        resized_size = 384
    elif pretrained_encoder == 'Resnet152':
        resized_size = 232
    img = np.array(Image.fromarray(img).resize((resized_size, resized_size)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    # prediction (beam search)
    if caption_model == 'show_tell':
        seq = decoder.beam_search(encoder_out, beam_size, word_map)
        return seq
    elif caption_model == 'att2all' or caption_model == 'spatial_att':
        seq, alphas = decoder.beam_search(encoder_out, beam_size, word_map)
        return seq, alphas
    elif caption_model == 'adaptive_att':
        seq, alphas, betas = decoder.beam_search(encoder_out, beam_size, word_map)
        return seq, alphas, betas


if __name__ == '__main__':
    output_path = "DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512"
    model_names = ["checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-0.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-1.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-2.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-3.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-4.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-5.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-6.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-7.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-8.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-9.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-10.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-11.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-12.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-13.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-14.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-15.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-16.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-17.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-18.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-19.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-20.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-21.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-22.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-23.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-24.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-25.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-26.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-27.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-28.pth.tar",
                   "checkpoint_DenseNet201_glove300_fine_tune_encoder_false_decoder_dim_512-epoch-29.pth.tar"
                   ]

    model_path = os.path.join(data_f, "output", output_path, "checkpoints")

    for model_name in model_names:
        checkpoint_path = os.path.join(model_path, model_name)  # model checkpoint
        beam_size = 5
        ifsmooth = False

        # load model
        checkpoint = torch.load(checkpoint_path, map_location=str(device))

        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()

        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()

        caption_model = checkpoint['caption_model']

        # encoder-decoder with beam search
        if caption_model == 'show_tell':
            seq = generate_caption(encoder, decoder, img, word_map, caption_model, beam_size)
            caption = [rev_word_map[ind] for ind in seq if
                       ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            print('Caption: ', ' '.join(caption))

        elif caption_model == 'att2all' or caption_model == 'spatial_att':
            seq, alphas = generate_caption(encoder, decoder, img, word_map, caption_model, beam_size)
            prediction = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            alphas = torch.FloatTensor(alphas)
            # visualize caption and attention of best sequence
            visualize_att(
                image_path=img,
                seq=seq,
                rev_word_map=rev_word_map,
                alphas=alphas,
                smooth=ifsmooth
            )

        elif caption_model == 'adaptive_att':
            seq, alphas, betas = generate_caption(encoder, decoder, img, word_map, caption_model, beam_size)
            alphas = torch.FloatTensor(alphas)
            visualize_att_beta(
                image_path=img,
                seq=seq,
                rev_word_map=rev_word_map,
                alphas=alphas,
                betas=betas,
                smooth=ifsmooth
            )
