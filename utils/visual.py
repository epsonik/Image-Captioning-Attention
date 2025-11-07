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
    smooth: bool = True,
    model_name: str = "model"
) -> None:
    """
    Visualize caption with weights and betas at every word.
    Saves a folder named {model_name}\_{image_stem} in the image directory containing:
      - the input image
      - one image per word with filename: {index}\_{word}.jpg

    Betas curve drawing is commented out.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # create output directory: head/{model_name}_{image_stem}
    head, tail = os.path.split(image_path)
    stem, ext = os.path.splitext(tail)
    out_dir = os.path.join(head, f"{model_name}_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    # save input image
    image.save(os.path.join(out_dir, f"0_input_{tail}"))

    # subplot settings
    num_col = len(words) - 1
    num_row = 1
    subplot_size = 4

    # graph settings
    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    img_size = 1
    fig_height = img_size
    fig_width = num_col + img_size

    grid = plt.GridSpec(fig_height, fig_width)

    # big image
    plt.subplot(grid[0: img_size, 0: img_size])
    plt.imshow(image)
    plt.axis('off')

    # Betas curve (commented out)
    # if betas is not None:
    #     plt.subplot(grid[0: fig_height - 1, img_size: fig_width])
    #
    #     x = range(1, len(words), 1)
    #     y = [(1 - betas[t].item()) for t in range(1, len(words))]
    #
    #     for a, b in zip(x, y):
    #         plt.text(a + 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    #
    #     plt.axis('off')
    #     plt.plot(x, y)

    for t in range(1, len(words)):
        if t > 50:
            break

        plt.subplot(grid[fig_height - 1, img_size + t - 1])
        # images
        plt.imshow(image)
        # words of sentence
        plt.title('%s' % (words[t]), color='black', fontsize=10,
                 horizontalalignment='center', verticalalignment='center')

        # alphas
        current_alpha = alphas[t, :]
        # handle torch tensors safely
        try:
            alpha_arr = current_alpha.detach().cpu().numpy()
        except Exception:
            alpha_arr = current_alpha.numpy()

        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha_arr, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(alpha_arr, [14 * 24, 14 * 24])

        plt.imshow(alpha, alpha=0.6)
        plt.set_cmap('jet')

        plt.axis('off')

        # --- save per-word overlay image ---
        # prepare heatmap from alpha (values 0..1)
        cmap = cm.get_cmap("jet")
        heatmap_rgba = (cmap(alpha)[:, :, :3] * 255).astype(np.uint8)  # drop alpha channel
        heat_img = Image.fromarray(heatmap_rgba).convert("RGBA")
        base_img = image.convert("RGBA")
        overlay = Image.blend(base_img, heat_img, alpha=0.6)

        # sanitize word for filename
        raw_word = words[t]
        safe_word = ''.join(c for c in raw_word if c.isalnum() or c in ('_', '-')).strip()
        if not safe_word:
            safe_word = "word"

        out_name = f"{t}_{safe_word}{ext}"
        overlay.save(os.path.join(out_dir, out_name))

    plt.savefig(os.path.join(out_dir, "att_" + tail), bbox_inches='tight')


def visualize_att(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    smooth: bool = True,
    model_name: str = "model"
) -> None:
    """
    Visualize caption with weights at every word.
    Saves a folder named {model_name}_{image_stem} in the image directory containing:
      - the input image
      - one image per word with filename: {index}_{word}{ext}
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # create output directory: head/{model_name}_{image_stem}
    head, tail = os.path.split(image_path)
    stem, ext = os.path.splitext(tail)
    out_dir = os.path.join(head, f"{model_name}_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    # save input image
    image.save(os.path.join(out_dir, f"0_input_{tail}"))

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

        # handle torch tensors safely
        try:
            alpha_arr = current_alpha.detach().cpu().numpy()
        except Exception:
            try:
                alpha_arr = current_alpha.numpy()
            except Exception:
                alpha_arr = np.array(current_alpha)

        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha_arr, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(alpha_arr, [14 * 24, 14 * 24])

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

        # --- save per-word overlay image ---
        cmap = cm.get_cmap("jet")
        heatmap_rgba = (cmap(alpha)[:, :, :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heatmap_rgba).convert("RGBA")
        base_img = image.convert("RGBA")
        overlay = Image.blend(base_img, heat_img, alpha=0.6)

        raw_word = words[t]
        safe_word = ''.join(c for c in raw_word if c.isalnum() or c in ('_', '-')).strip()
        if not safe_word:
            safe_word = "word"

        out_name = f"{t}_{safe_word}{ext}"
        overlay.save(os.path.join(out_dir, out_name))

    plt.savefig(os.path.join(out_dir, "att_" + tail), bbox_inches='tight')
