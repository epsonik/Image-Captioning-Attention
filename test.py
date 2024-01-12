"""
Compute the correct BLEU, CIDEr, ROUGE and METEOR scores for a checkpoint on
the validation or test sets without Teacher Forcing.
"""
import csv
import json
import os

from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import CaptionDataset, load_checkpoint
from metrics import Metrics
from config import config

device = torch.device(
    config.cuda_device if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
data_f = os.path.join(config.base_path, "data")
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = os.path.join(data_f, "evaluation", 'wordmap' + '.json')

# load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

vocab_size = len(word_map)

# create ix2word map
rev_word_map = {v: k for k, v in word_map.items()}

# normalization transform
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def evaluate(encoder, decoder, caption_model, beam_size: int) -> float:
    """
    Parameters
    ----------
    beam_size : int
        Beam size at which to generate captions for evaluation. Set beam_size
        to 1 if you want to use greedy search.

    Returns
    -------
    bleu4 : float
        BLEU-4 score
    """
    loader = DataLoader(
        CaptionDataset(
            os.path.join(data_f, "evaluation"), data_name, 'test',
            transform=transforms.Compose([normalize])
        ),
        # TODO: batched beam search. Therefore, DO NOT use a batch_size greater
        # than 1 - IMPORTANT!
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    # store ground truth captions and predicted captions (word id) of each image
    # for n images, each of them has one prediction and multiple ground truths (a, b, c...):
    # prediction = [ [pred1], [pred2], ..., [predn] ]
    # ground_truth = [ [ [gt1a], [gt1b], [gt1c] ], ..., [ [gtna], [gtnb] ] ]
    ground_truth = list()
    prediction = list()

    # for each image
    for i, (image, caps, caplens, allcaps) in enumerate(
        tqdm(loader, desc="Evaluating at beam size " + str(beam_size))):
        # move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # forward encoder
        encoder_out = encoder(image)

        # ground_truth
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        ground_truth.append(img_captions)

        # prediction (beam search)
        if caption_model == 'show_tell':
            seq = decoder.beam_search(encoder_out, beam_size, word_map)
        elif caption_model == 'att2all' or caption_model == 'spatial_att':
            seq, _ = decoder.beam_search(encoder_out, beam_size, word_map)
        elif caption_model == 'adaptive_att':
            seq, _, _ = decoder.beam_search(encoder_out, beam_size, word_map)

        pred = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        prediction.append(pred)

        assert len(ground_truth) == len(prediction)

    # calculate metrics
    metrics = Metrics(ground_truth, prediction, rev_word_map)
    scores = metrics.all_metrics

    return scores


def generate_report(report_name, config_name, bleu1, bleu2, bleu3, bleu4, cider, rouge):
    """
    Method to generate summary of the test results. Made from files in the results directory.

    Parameters
    ----------
    results_path: str
        Path to the results directory
    Returns
    -------
        CSV file with summary of the results.

    """
    # Names of the evaluation metrics
    header = ["config_name", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"]

    temp = dict()
    temp["config_name"] = config_name
    temp["Bleu_1"] = bleu1
    temp["Bleu_2"] = bleu2
    temp["Bleu_3"] = bleu3
    temp["Bleu_4"] = bleu4
    temp["ROUGE_L"] = rouge
    temp["CIDEr"] = cider
    # Save final csv file

    with open(os.path.join(data_f, report_name), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(temp)
        f.close()


if __name__ == '__main__':
    # 1
    output_path = ["Resnet101_glove300_fte_true_decoder_dim_512", "Resnet101_glove300_fte_false_decoder_dim_512"]
    # 2
    # output_path = ["DenseNet201_glove300_fte_true_decoder_dim_256", "DenseNet201_glove300_fte_false_decoder_dim_256"]
    # # 3
    # output_path = ["DenseNet201_glove300_fte_true", "DenseNet201_glove300_fte_false",
    #                "Resnet101_glove300_fte_true", "Resnet101_glove300_fte_false",
    #                "DenseNet201_glove300_fte_true_decoder_dim_512", "Resnet101_glove300_fte_false_decoder_dim_256",
    #                "DenseNet201_glove300_fte_false_decoder_dim_512", "Resnet101_glove300_fte_true_decoder_dim_256"
    #                ]
    for data_name in output_path:
        # path to save checkpoints
        model_path = os.path.join(data_f, "output/dnt", data_name, "checkpoints")
        checkpoint = os.path.join(model_path, 'best_checkpoint_' + data_name + '.pth.tar')  # model checkpoint
        print(checkpoint)
        beam_size = 1
        # load model
        checkpoint = torch.load(checkpoint, map_location=str(device))

        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()

        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()

        caption_model = checkpoint['caption_model']


        def temp(beam_size, report_name):
            print("Scores for ", data_name)
            (bleu1, bleu2, bleu3, bleu4), cider, rouge = evaluate(encoder, decoder, config.caption_model, beam_size)

            print("\nScores @ beam size of %d are:" % beam_size)
            print("   BLEU-1: %.4f" % bleu1)
            print("   BLEU-2: %.4f" % bleu2)
            print("   BLEU-3: %.4f" % bleu3)
            print("   BLEU-4: %.4f" % bleu4)
            print("   CIDEr: %.4f" % cider)
            print("   ROUGE-L: %.4f" % rouge)

            generate_report(report_name, data_name, bleu1, bleu2, bleu3, bleu4, cider, rouge)


        temp(1, "final_results_k1.csv")
        # temp(2, "final_results_k2.csv")
        # temp(5, "final_results_k5.csv")
        # temp(8, "final_results_k8.csv")
