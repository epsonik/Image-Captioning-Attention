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
        Name of the model, used for creating the output directory.

    smooth : bool, optional, default=True
        Smooth weights or not?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # Create directory
    clean_model_name = model_name.replace('best_checkpoint', '').replace('checkpoint', '').replace('.pth.tar', '')
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(clean_model_name, basename)
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    image.save(os.path.join(output_dir, 'original.png'))

    # Save the main visualization figure
    num_col = len(words) - 1
    subplot_size = 4
    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * 4)
    img_size = 4
    fig_height = img_size
    fig_width = num_col + img_size
    grid = plt.GridSpec(fig_height, fig_width)
    plt.subplot(grid[0: img_size, 0: img_size])
    plt.imshow(image)
    plt.axis('off')
    if betas is not None:
        plt.subplot(grid[0: fig_height - 1, img_size: fig_width])
        x = range(1, len(words), 1)
        y = [(1 - betas[t].item()) for t in range(1, len(words))]
        for a, b in zip(x, y):
            plt.text(a + 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)
        plt.axis('off')
        plt.plot(x, y)
    for t in range(1, len(words)):
        if t > 50: break
        ax = plt.subplot(grid[fig_height - 1, img_size + t - 1])
        ax.imshow(image)
        ax.set_title(words[t], fontsize=10)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        ax.imshow(alpha, alpha=0.6, cmap='jet')
        ax.axis('off')
    fig.savefig(os.path.join(output_dir, f'att_beta_{basename}.png'))
    plt.close(fig)

    # Save individual attention images
    for t in range(1, len(words)):
        if t > 50:
            break

        fig_att, ax_att = plt.subplots()
        ax_att.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        ax_att.imshow(alpha, alpha=0.6, cmap='jet')
        ax_att.axis('off')

        word = words[t]
        ax_att.set_title(word, y=-0.2, fontsize=12)
        filename = f"{t}_{word}.png"
        fig_att.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
        plt.close(fig_att)


def visualize_att(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    model_name: str,
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

    model_name : str
        Name of the model, used for creating the output directory.

    smooth : bool, optional, default=True
        Smooth weights or not?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # Create directory
    clean_model_name = model_name.replace('best_checkpoint', '').replace('checkpoint', '').replace('.pth.tar', '')
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(clean_model_name, basename)
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    image.save(os.path.join(output_dir, 'original.png'))

    # Save individual attention images
    for t in range(len(words)):
        if t > 50:
            break

        fig, ax = plt.subplots()
        ax.imshow(image)

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            ax.imshow(alpha, alpha=0, cmap=cm.Greys_r)
        else:
            ax.imshow(alpha, alpha=0.8, cmap=cm.Greys_r)

        ax.axis('off')

        word = words[t]
        ax.set_title(word, y=-0.2, fontsize=12)
        # Sanitize word for filename
        sanitized_word = "".join(c for c in word if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"{t}_{sanitized_word}.png"

        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
