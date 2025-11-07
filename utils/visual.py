import os
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image
import numpy as np


def visualize_att_beta(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    betas: list,
    model_name: str,
    smooth: bool = True
) -> None:
    """
    Visualize caption with weights and betas at every word.

    Parameters
    ----------
    image_path : str
        Path to image that has been captioned

    seq : list
        Generated caption on the above mentioned image using beam search

    alphas : list
        Attention weights at each time step

    betas : list
        Sentinel gate at each time step (only in 'adaptive_att' mode)

    rev_word_map : Dict[int, str]
        Reverse word mapping, i.e. ix2word

    model_name : str
        Name of the model, used for creating the output directory

    smooth : bool, optional, default=True
        Smooth weights or not?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # Create output directory
    head, tail = os.path.split(image_path)
    img_name = tail.split('.')[0]
    output_dir = os.path.join(head, f"{model_name}_{img_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    image.save(os.path.join(output_dir, tail))


    for t in range(1, len(words)):
        if t > 50:
            break

        # Create a new figure and axes for each word
        fig, ax = plt.subplots()
        ax.imshow(image)

        # alphas
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        ax.imshow(alpha, alpha=0.6, cmap='jet')

        ax.axis('off')
        # Remove padding and margins
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Save the figure for the current word
        word = words[t]
        # Sanitize word for filename
        sanitized_word = "".join(c if c.isalnum() else "_" for c in word)
        filename = f"{t}_{sanitized_word}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close(fig)


def visualize_att(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    smooth: bool = True
) -> None:
    """
    Visualize caption with weights at every word.

    Adapted from: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    Parameters
    ----------
    image_path : str
        Path to image that has been captioned

    seq : list
        Generated caption on the above mentioned image using beam search

    alphas : list
        Attention weights at each time step

    rev_word_map : Dict[int, str]
        Reverse word mapping, i.e. ix2word

    smooth : bool, optional, default=True
        Smooth weights or not?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # subplot settings
    num_col = 5
    num_row = int(np.ceil(len(words) / float(num_col)))
    subplot_size = 4

    # graph settings
    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    for t in range(len(words)):
        if t > 50:
            break

        plt.subplot(num_row, num_col, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)

        plt.imshow(image)

        current_alpha = alphas[t, :]

        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    head, tail = os.path.split(image_path)
    plt.savefig(os.path.join(head, "att_" + tail), bbox_inches='tight')
