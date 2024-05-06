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

from utils import CaptionDataset
from metrics import Metrics
import pathlib

device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu")
data_f = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = os.path.join(data_f,
                             "output/show_tell_InceptionV3_decoder_dim_1024_fine_tune_encoder_true_fine_tune_embeddings_true/wordmap.json")

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
            os.path.join(data_f,
                         'output/show_tell_InceptionV3_decoder_dim_1024_fine_tune_encoder_true_fine_tune_embeddings_true'),
            data_name, 'test',
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
    img_paths = list()

    # for each image
    for i, (image, caps, caplens, allcaps, img_path) in enumerate(
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
        img_paths.append(img_path)
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
    cocoEvalObj = Metrics(ground_truth, prediction, rev_word_map, img_paths)
    cocoEvalObj.img_to_eval()

    calculated_metrics = {}
    # Store metrics  values in dictionary by metrics names
    for metric, score in cocoEvalObj.eval.items():
        calculated_metrics[metric] = score

    print("Calculating final results")
    imgToEval = cocoEvalObj.imgToEval

    model_path = os.path.join(data_f, "results", data_name, 'k-' + str(beam_size))
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    final_model_name = model_name.replace(".pth.tar", '-k-' + str(beam_size))

    evaluation_results_save_path = os.path.join(model_path,
                                                final_model_name + '.json')
    with open(evaluation_results_save_path, 'w') as outfile:
        json.dump(
            {'overall': calculated_metrics, 'dataset_name': final_model_name, 'imgToEval': imgToEval},
            outfile)

    return cocoEvalObj.all_metrics


def generate_report(report_name, config_name, beam_size, bleu1, bleu2, bleu3, bleu4, cider, rouge):
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

    model_path = os.path.join(data_f, "results", data_name, 'k-' + str(beam_size))
    evaluation_results_save_path = os.path.join(model_path, report_name)
    with open(evaluation_results_save_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(temp)
        f.close()


def generate_report_for_all_models(results_path):
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
    header = ["config_name", "loss", "epoch", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr",
              "SPICE", "WMD"]
    print(f'\n Final results saved to final_results.csv')
    all_results = []
    # iterate over all files in results directory
    for x in os.listdir(results_path):
        # use just .json files
        if x.endswith(".json"):
            # Load data from file with results particular for configuaration
            results_for_report = json.load(open(os.path.join(results_path, x), 'r'))
            # Add column with the configuration name to name the specific results.
            config_name = x.replace(".json", '')
            b = config_name.split("-")
            results_for_report["overall"]["config_name"] = config_name
            results_for_report["overall"]["loss"] = b[2]
            results_for_report["overall"]["epoch"] = b[1]
            # Save the results to the table to save it in the next step
            all_results.append(results_for_report["overall"])
    # Save final csv file

    with open(results_path + "/final_results.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)


if __name__ == '__main__':

    configs = dict()
    output_path2 = [
        "best_checkpoint_show_tell_InceptionV3_decoder_dim_1024_fine_tune_encoder_true_fine_tune_embeddings_true-epoch-59.pth.tar"]
    output_path = ["show_tell_InceptionV3_decoder_dim_1024_fine_tune_encoder_true_fine_tune_embeddings_true"]
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    for data_name in output_path:
        # path to save checkpoints
        model_path = os.path.join(data_f, "output", data_name, "checkpoints")
        # checkpoint = os.path.join(model_path, 'checkpoint_' + data_name + '.pth.tar')  # model checkpoint
        for model_name in output_path2:
            checkpoint = os.path.join(model_path, model_name)  # model checkpoint
            print(checkpoint)
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
                (bleu1, bleu2, bleu3, bleu4), cider, rouge = evaluate(encoder, decoder, caption_model, beam_size)

                print("\nScores @ beam size of %d are:" % beam_size)
                print("   BLEU-1: %.4f" % bleu1)
                print("   BLEU-2: %.4f" % bleu2)
                print("   BLEU-3: %.4f" % bleu3)
                print("   BLEU-4: %.4f" % bleu4)
                print("   CIDEr: %.4f" % cider)
                print("   ROUGE-L: %.4f" % rouge)

                # generate_report(report_name, data_name, bleu1, bleu2, bleu3, bleu4, cider, rouge)
                generate_report(report_name, model_name, beam_size, bleu1, bleu2, bleu3, bleu4, cider, rouge)


            temp(1, "final_results_k1.csv")
            temp(2, "final_results_k2.csv")
            temp(5, "final_results_k5.csv")
            temp(8, "final_results_k8.csv")
