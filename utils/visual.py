from typing import Dict
import os
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
    model_name: str = "",
    smooth: bool = True,
) -> None:
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # subplot settings
    num_col = max(1, len(words) - 1)
    num_row = 1
    subplot_size = 4

    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    img_size = 1
    fig_height = img_size
    fig_width = num_col + img_size

    grid = plt.GridSpec(fig_height, fig_width)

    # big image
    ax_img = plt.subplot(grid[0: img_size, 0: img_size])
    ax_img.imshow(image)
    ax_img.axis('off')

    if betas is not None and len(words) > 1:
        ax_beta = plt.subplot(grid[0: fig_height, img_size: fig_width])
        x = list(range(1, len(words)))
        y = [(1 - betas[t].item()) for t in range(1, len(words))]
        ax_beta.plot(x, y)
        for a, b in zip(x, y):
            ax_beta.text(a + 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)
        ax_beta.axis('off')

    for t in range(1, len(words)):
        if t > 50:
            break

        ax = plt.subplot(grid[fig_height - 1, img_size + t - 1])
        ax.imshow(image)
        ax.axis('off')

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        ax.imshow(alpha, alpha=0.6, cmap='jet')

        # place the word below the image using axis-relative coordinates
        ax.text(
            0.5, -0.12, words[t],
            transform=ax.transAxes,
            ha='center', va='top',
            color='black', backgroundcolor='white', fontsize=10
        )

    head, tail = os.path.split(image_path)
    # build cleaned model name
    if model_name:
        cleaned = model_name.replace('best_checkpoint_', '').replace('.pth.tar', '')
        out_name = f"{cleaned}_{tail}"
    else:
        out_name = tail
    out_path = os.path.join(head, out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def visualize_att(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    model_name: str = "",
    smooth: bool = True,
) -> None:
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # subplot settings
    num_col = 5
    num_row = int(np.ceil(len(words) / float(num_col)))
    subplot_size = 4

    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    for t in range(len(words)):
        if t > 50:
            break

        ax = plt.subplot(num_row, num_col, t + 1)

        ax.imshow(image)

        current_alpha = alphas[t, :]

        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            ax.imshow(alpha, alpha=0)
        else:
            ax.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        ax.axis('off')

        # place the word below the image using axis-relative coordinates
        ax.text(
            0.5, -0.12, words[t],
            transform=ax.transAxes,
            ha='center', va='top',
            color='black', backgroundcolor='white', fontsize=12
        )

    head, tail = os.path.split(image_path)
    if model_name:
        cleaned = model_name.replace('best_checkpoint_', '').replace('.tar.tgz', '')
        out_name = f"{cleaned}_{tail}"
    else:
        out_name = tail
    out_path = os.path.join(head, out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
