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
    smooth: bool = True
) -> None:
    """
    Visualize caption with weights and betas at every word.
    Fixed: ensure full image is shown in each subplot by using a numpy array,
    explicit extent and aspect='auto' for both image and alpha overlays.
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    words = [rev_word_map[ind] for ind in seq]

    num_col = len(words) - 1
    num_row = 1
    subplot_size = 4

    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    img_size = 1
    fig_height = img_size
    fig_width = num_col + img_size

    grid = plt.GridSpec(fig_height, fig_width)

    # big image
    ax = plt.subplot(grid[0: img_size, 0: img_size])
    ax.imshow(image_np, aspect='auto', extent=(0, w, h, 0))
    ax.axis('off')

    if betas is not None:
        axb = plt.subplot(grid[0: fig_height - 1, img_size: fig_width])
        x = range(1, len(words), 1)
        y = [(1 - betas[t].item()) for t in range(1, len(words))]

        for a, b in zip(x, y):
            axb.text(a + 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)

        axb.axis('off')
        axb.plot(x, y)

    for t in range(1, len(words)):
        if t > 50:
            break

        ax = plt.subplot(grid[fig_height - 1, img_size + t - 1])
        ax.imshow(image_np, aspect='auto', extent=(0, w, h, 0))
        ax.set_title('%s' % (words[t]), color='black', fontsize=10,
                     horizontalalignment='center', verticalalignment='center')

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        # overlay with same extent so it covers whole image
        ax.imshow(alpha, alpha=0.6, cmap='jet', extent=(0, w, h, 0), interpolation='bilinear')
        ax.axis('off')

    plt.tight_layout()
    head, tail = os.path.split(image_path)
    plt.savefig(os.path.join(head, "att_" + tail), bbox_inches='tight')


def visualize_att(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    smooth: bool = True
) -> None:
    """
    Visualize caption with weights at every word.
    Fixed: ensure full image is shown in each subplot by using a numpy array,
    explicit extent and aspect='auto' for both image and alpha overlays.
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    words = [rev_word_map[ind] for ind in seq]

    num_col = 5
    num_row = int(np.ceil(len(words) / float(num_col)))
    subplot_size = 4

    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    for t in range(len(words)):
        if t > 50:
            break

        ax = plt.subplot(num_row, num_col, t + 1)
        ax.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        ax.imshow(image_np, aspect='auto', extent=(0, w, h, 0))

        current_alpha = alphas[t, :]

        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            ax.imshow(alpha, alpha=0, cmap=cm.Greys_r, extent=(0, w, h, 0), interpolation='bilinear')
        else:
            ax.imshow(alpha, alpha=0.8, cmap=cm.Greys_r, extent=(0, w, h, 0), interpolation='bilinear')

        ax.axis('off')

    plt.tight_layout()
    head, tail = os.path.split(image_path)
    plt.savefig(os.path.join(head, "att_" + tail), bbox_inches='tight')
