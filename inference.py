import json
import os
import numpy as np
import argparse
import re
from typing import Dict, List, Tuple, Optional
# from scipy.misc import imread, imresize
from imageio import imread
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import visualize_att_beta, visualize_att
from metrics import Metrics
from metrics.bleu import Bleu
from metrics.meteor import Meteor
from metrics.rouge import Rouge
from metrics.cider import Cider

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def generate_caption(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: str,
    word_map: Dict[str, int],
    caption_model: str,
    beam_size: int = 3
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
    img = np.array(Image.fromarray(img).resize((256, 256)))
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


def load_coco_annotations(annotations_path: str) -> Dict[int, List[str]]:
    """
    Load COCO annotations from JSON file.

    Parameters
    ----------
    annotations_path : str
        Path to COCO annotations JSON file (e.g., captions_val2014.json)

    Returns
    -------
    Dict[int, List[str]]
        Dictionary mapping image_id to list of caption strings
    """
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from image_id to list of captions
    image_captions = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']

        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)

    return image_captions


def extract_image_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract COCO image ID from filename.

    COCO filenames follow the pattern: COCO_val2014_000000XXXXXX.jpg
    or COCO_train2014_000000XXXXXX.jpg

    Parameters
    ----------
    filename : str
        Image filename

    Returns
    -------
    Optional[int]
        Image ID if found, None otherwise
    """
    # Extract just the filename without path
    basename = os.path.basename(filename)

    # Try to match COCO filename pattern
    match = re.search(r'COCO_(?:val|train)2014_(\d+)', basename)
    if match:
        return int(match.group(1))

    # If no match, try to extract any sequence of digits
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))

    return None


def evaluate_caption(
    generated_caption: str,
    reference_captions: List[str]
) -> Dict[str, float]:
    """
    Evaluate a generated caption against reference captions using multiple metrics.

    Parameters
    ----------
    generated_caption : str
        Generated caption text
    reference_captions : List[str]
        List of reference caption texts

    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics (BLEU-1 to BLEU-4, METEOR, ROUGE-L, CIDEr)
    """
    # Prepare data in the format expected by the metrics
    # Reference: list of reference captions for this image
    # Hypothesis: list containing the generated caption
    references = [reference_captions]
    hypotheses = [[generated_caption]]

    results = {}

    # Calculate BLEU scores (1-4)
    bleu_scorer = Bleu(n=4)
    bleu_scores, _ = bleu_scorer.compute_score(references, hypotheses)
    results['BLEU-1'] = bleu_scores[0]
    results['BLEU-2'] = bleu_scores[1]
    results['BLEU-3'] = bleu_scores[2]
    results['BLEU-4'] = bleu_scores[3]

    # Calculate METEOR score
    try:
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(references, hypotheses)
        results['METEOR'] = meteor_score
    except Exception as e:
        print(f"Warning: Could not calculate METEOR score: {e}")
        results['METEOR'] = 0.0

    # Calculate ROUGE-L score
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(references, hypotheses)
    results['ROUGE-L'] = rouge_score

    # Calculate CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, hypotheses)
    results['CIDEr'] = cider_score

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions for images and optionally evaluate them')
    parser.add_argument('--image_folder', type=str,
                        default='/home/bartosiewicz/mateusz/Image-Captioning-Attention/assets/make-weights',
                        help='Path to folder containing images')
    parser.add_argument('--model_folder', type=str,
                        default='/home/bartosiewicz/mateusz/Image-Captioning-Attention/data/output/t',
                        help='Path to folder containing model checkpoints')
    parser.add_argument('--wordmap_path', type=str,
                        default='/home/bartosiewicz/mateusz/Image-Captioning-Attention/data/evaluation/wordmap.json',
                        help='Path to word map JSON file')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size for beam search')
    parser.add_argument('--smooth', action='store_true',
                        help='Enable smooth visualization')
    parser.add_argument('--evaluate', action='store_true',
                        help='Enable evaluation mode (requires --annotations_path)')
    parser.add_argument('--annotations_path', type=str, default=None,
                        help='Path to COCO annotations JSON file (e.g., annotations/captions_val2014.json)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save evaluation results JSON file')

    args = parser.parse_args()

    image_folder = args.image_folder
    model_folder = args.model_folder
    wordmap_path = args.wordmap_path
    beam_size = args.beam_size
    ifsmooth = args.smooth

    # Load annotations if evaluation is enabled
    image_captions = None
    if args.evaluate:
        if args.annotations_path is None:
            print("Error: --annotations_path is required when --evaluate is enabled")
            exit(1)
        if not os.path.exists(args.annotations_path):
            print(f"Error: Annotations file not found at {args.annotations_path}")
            exit(1)

        print(f"Loading COCO annotations from {args.annotations_path}...")
        image_captions = load_coco_annotations(args.annotations_path)
        print(f"Loaded annotations for {len(image_captions)} images")

    # load word map (word2ix)
    with open(wordmap_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth.tar')]
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Store all evaluation results
    all_results = []

    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        print(f"\nProcessing image: {image_file}")

        # Extract image ID for evaluation
        image_id = None
        if args.evaluate:
            image_id = extract_image_id_from_filename(image_file)
            if image_id is not None:
                print(f"  Extracted image ID: {image_id}")
            else:
                print(f"  Warning: Could not extract image ID from filename: {image_file}")

        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            print(f"  Using model: {model_file}")

            # load model
            checkpoint = torch.load(model_path, map_location=str(device))

            decoder = checkpoint['decoder']
            decoder = decoder.to(device)
            decoder.eval()

            encoder = checkpoint['encoder']
            encoder = encoder.to(device)
            encoder.eval()

            caption_model = checkpoint['caption_model']

            # encoder-decoder with beam search
            generated_caption_text = None

            if caption_model == 'show_tell':
                seq = generate_caption(encoder, decoder, img_path, word_map, caption_model, beam_size)
                caption = [rev_word_map[ind] for ind in seq if
                           ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                generated_caption_text = ' '.join(caption)
                print(f'    Generated Caption: {generated_caption_text}')

            elif caption_model == 'att2all' or caption_model == 'spatial_att':
                seq, alphas = generate_caption(encoder, decoder, img_path, word_map, caption_model, beam_size)
                caption = [rev_word_map[ind] for ind in seq if
                           ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                generated_caption_text = ' '.join(caption)
                print(f'    Generated Caption: {generated_caption_text}')

                alphas = torch.FloatTensor(alphas)
                # visualize caption and attention of best sequence
                visualize_att(
                    image_path=img_path,
                    seq=seq,
                    rev_word_map=rev_word_map,
                    alphas=alphas,
                    model_name=model_file,
                    smooth=ifsmooth
                )

            elif caption_model == 'adaptive_att':
                seq, alphas, betas = generate_caption(encoder, decoder, img_path, word_map, caption_model, beam_size)
                caption = [rev_word_map[ind] for ind in seq if
                           ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                generated_caption_text = ' '.join(caption)
                print(f'    Generated Caption: {generated_caption_text}')

                alphas = torch.FloatTensor(alphas)
                visualize_att_beta(
                    image_path=img_path,
                    seq=seq,
                    rev_word_map=rev_word_map,
                    alphas=alphas,
                    betas=betas,
                    model_name=model_file,
                    smooth=ifsmooth
                )

            # Evaluate if enabled
            if args.evaluate and generated_caption_text is not None and image_id is not None:
                if image_id in image_captions:
                    reference_captions = image_captions[image_id]
                    print(f"    Reference captions ({len(reference_captions)}):")
                    for ref_cap in reference_captions:
                        print(f"      - {ref_cap}")

                    # Calculate metrics
                    metrics = evaluate_caption(generated_caption_text, reference_captions)

                    print(f"    Evaluation Metrics:")
                    print(f"      BLEU-1: {metrics['BLEU-1']:.4f}")
                    print(f"      BLEU-2: {metrics['BLEU-2']:.4f}")
                    print(f"      BLEU-3: {metrics['BLEU-3']:.4f}")
                    print(f"      BLEU-4: {metrics['BLEU-4']:.4f}")
                    print(f"      METEOR: {metrics['METEOR']:.4f}")
                    print(f"      ROUGE-L: {metrics['ROUGE-L']:.4f}")
                    print(f"      CIDEr: {metrics['CIDEr']:.4f}")

                    # Store results
                    result_entry = {
                        'image_file': image_file,
                        'image_id': image_id,
                        'model': model_file,
                        'generated_caption': generated_caption_text,
                        'reference_captions': reference_captions,
                        'metrics': metrics
                    }
                    all_results.append(result_entry)
                else:
                    print(f"    Warning: No reference captions found for image ID {image_id}")

    # Save results if requested
    if args.save_results and all_results:
        print(f"\nSaving evaluation results to {args.save_results}")
        with open(args.save_results, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved successfully!")

    # Print summary if evaluation was performed
    if args.evaluate and all_results:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        # Calculate average metrics across all images
        avg_metrics = {
            'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0,
            'METEOR': 0.0, 'ROUGE-L': 0.0, 'CIDEr': 0.0
        }

        for result in all_results:
            for metric_name, value in result['metrics'].items():
                avg_metrics[metric_name] += value

        num_results = len(all_results)
        for metric_name in avg_metrics:
            avg_metrics[metric_name] /= num_results

        print(f"Average metrics across {num_results} images:")
        print(f"  BLEU-1:  {avg_metrics['BLEU-1']:.4f}")
        print(f"  BLEU-2:  {avg_metrics['BLEU-2']:.4f}")
        print(f"  BLEU-3:  {avg_metrics['BLEU-3']:.4f}")
        print(f"  BLEU-4:  {avg_metrics['BLEU-4']:.4f}")
        print(f"  METEOR:  {avg_metrics['METEOR']:.4f}")
        print(f"  ROUGE-L: {avg_metrics['ROUGE-L']:.4f}")
        print(f"  CIDEr:   {avg_metrics['CIDEr']:.4f}")
        print("=" * 80)
