import json
from typing import List, Tuple, Dict, Union
import numpy as np

from .bleu import Bleu
from .cider import Cider
from .rouge import Rouge


class Metrics:
    """
    Compute metrics on given reference and candidate sentences set. Now supports
    BLEU, CIDEr, METEOR and ROUGE-L.

    Parameters
    ----------
    references : List[List[List[int]]] ([[[ref1a], [ref1b], [ref1c]], ..., [[refna], [refnb]]])
        Reference sencences (list of word ids)

    candidates : List[List[int]] ([[hyp1], [hyp2], ..., [hypn]]):
        Candidate sencences (list of word ids)

    rev_word_map : Dict[int, str]
        ix2word map
    """

    def __init__(
        self,
        references: List[List[List[int]]],
        candidates: List[List[int]],
        rev_word_map: Dict[int, str],
        img_paths: List[str]
    ) -> None:
        self.eval = {}
        corpus = setup_corpus(references, candidates, rev_word_map)
        self.ref_sentence = corpus[0]
        self.hypo_sentence = corpus[1]
        self.img_paths = img_paths
        self.imgToEval = {}

    @property
    def bleu(self) -> Tuple[List[float], List[List[float]]]:
        bleu_score = Bleu().compute_score(self.ref_sentence, self.hypo_sentence)
        return bleu_score

    @property
    def cider(self) -> Tuple[np.float64, np.ndarray]:
        cider_score = Cider().compute_score(self.ref_sentence, self.hypo_sentence)
        return cider_score

    @property
    def rouge(self) -> Tuple[np.float64, np.ndarray]:
        rouge_score = Rouge().compute_score(self.ref_sentence, self.hypo_sentence)
        return rouge_score

    @property
    def all_metrics(self) -> Tuple[Tuple[float, float, float, float], np.float64, np.float64]:
        """Return all metrics"""
        return (self.bleu[0][0], self.bleu[0][0], self.bleu[0][2], self.bleu[0][3]), self.cider[0], self.rouge[0]

    def img_to_eval(self):
        scorers = [
            (Bleu(), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr"),
            (Rouge(), "ROUGE_L"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.ref_sentence, self.hypo_sentence)
            if type(method) == list:
                for sc, scs, method_name in zip(score, scores, method):
                    self.setEval(sc, method_name)
                    self.setImgToEvalImgs(scs, self.img_paths, method_name)
                    print("%s: %0.3f" % (method_name, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, self.img_paths, method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in list(self.imgToEval.items())]

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score, ref_sentence, hypo_sentence in zip(imgIds, scores, self.ref_sentence, self.hypo_sentence):
            imgId = imgId[0]
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
                self.imgToEval[imgId]["ground_truth_captions"] = ref_sentence
                self.imgToEval[imgId]["captions"] = hypo_sentence
            self.imgToEval[imgId][method] = score


def setup_corpus(
    references: List[List[List[int]]],
    candidates: List[List[int]],
    rev_word_map: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    ref_sentence = []
    hypo_sentence = []

    for cnt, each_image in enumerate(references):
        # ground truths
        cur_ref_sentence = []
        for cap in each_image:
            sentence = [rev_word_map[ix] for ix in cap]
            cur_ref_sentence.append(' '.join(sentence))

        ref_sentence.append(cur_ref_sentence)

        # predictions
        sentence = [rev_word_map[ix] for ix in candidates[cnt]]
        hypo_sentence.append([' '.join(sentence)])
    return ref_sentence, hypo_sentence
