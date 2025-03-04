from typing import List

import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
bertscore = evaluate.load("bertscore")


def context_recall(ground_truth: str, contexts: List[str])->float:
    """
    Calc rouge btw contexts and ground truth.
    Interpretation: ngram match (recall) btw contexts and desired answer.

    ROUGE - https://huggingface.co/spaces/evaluate-metric/rouge

    return: average rouge for all contexts.
    """
    rs = []
    for c in contexts:
        rs.append(
            rouge.compute(
                predictions=[str(c)],
                references=[str(ground_truth)],
            )["rouge1"]  # Меняем на rouge1
        )
    return np.mean(rs)


def context_precision(ground_truth: str, contexts: List[str])->float:
    """
    Calc blue btw contexts and ground truth.
    Interpretation: ngram match (precision) btw contexts and desired answer.

    BLEU - https://aclanthology.org/P02-1040.pdf
    max_order - max n-grams to count

    return: average bleu (precision2, w/o brevity penalty) for all contexts.
    """
    bs = []
    for c in contexts:

        try:
            bs.append(
                bleu.compute(
                    predictions=[str(c)],
                    references=[str(ground_truth)],
                    max_order=1,
                )["precisions"][0]
            )
        except ZeroDivisionError:
            bs.append(0)

    return np.mean(bs)


def answer_correctness_literal(
    ground_truth: str,
    answer: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 1,
)->float:
    """
    Calc chrF btw answer and ground truth.
    Interpretation: lingustic match btw answer and desired answer.

    chrF - https://aclanthology.org/W15-3049.pdf
    char_order - n-gram length for chars, default is 6 (from the article)
    word_order - n-gram length for words (chrF++), default is 2 (as it outperforms simple chrF)
    beta - recall weight, beta=1 - simple F1-score

    return: chrF for answ and gt.
    """

    score = chrf.compute(
        predictions=[str(answer)],
        references=[str(ground_truth)],
        word_order=word_order,
        char_order=char_order,
        beta=beta,
    )["score"]

    return score/100


def answer_correctness_neural(
    ground_truth: str,
    answer: str,
    model_type: str = "cointegrated/rut5-base",
)->float:
    """
    Calc bertscore btw answer and ground truth.
    Interpretation: semantic cimilarity btw answer and desired answer.

    BertScore - https://arxiv.org/pdf/1904.09675.pdf
    model_type - embeds model  (default t5 as the best from my own research and experience)

    return: bertscore-f1 for answ and gt.
    """

    score = bertscore.compute(
        predictions=[str(answer)],
        references=[str(ground_truth)],
        batch_size=1,
        model_type=model_type,
        num_layers=11,
    )["f1"]

    return score


class ValidatorSimple:
    """
    Расчет простых метрик качества для заданного датасета.
    """
    def __init__(
        self,
        neural: bool = False,
    ):
        """
        param neural: есть гпу или нет. По дефолту ее нет(
        """
        self.neural = neural

    def score_sample(
        self,
        answer: str,
        ground_truth: str,
        context: List[str],
    ):
        """
        Расчет для конкретного сэмпла в тестовом датасете.
        """
        scores = {}
        scores["context_recall"] = [
            context_recall(
                ground_truth,
                context,
            )
        ]
        scores["context_precision"] = [
            context_precision(
                ground_truth,
                context,
            )
        ]
        scores["answer_correctness_literal"] = [
            answer_correctness_literal(
                ground_truth=ground_truth,
                answer=answer,
            )
        ]
        if self.neural:
            scores["answer_correctness_neural"] = [
                answer_correctness_neural(
                    ground_truth=ground_truth,
                    answer=answer,
                )
            ]
        return scores

    def validate_rag(
        self,
        test_set: pd.DataFrame,
        threshold_good: float = 0.8,
        threshold_mid: float = 0.5,
    ):
        """
        param test_set: пандас датасет с нужными полями: answer, ground_truth, context, question
        """
        res = {}
        good_models_count = {
            "context_recall": 0,
            "context_precision": 0,
            "answer_correctness_literal": 0,
            "answer_correctness_neural": 0,
        }
        mid_models_count = {
            "context_recall": 0,
            "context_precision": 0,
            "answer_correctness_literal": 0,
            "answer_correctness_neural": 0,
        }

        bad_models_count = {
            "context_recall": 0,
            "context_precision": 0,
            "answer_correctness_literal": 0,
            "answer_correctness_neural": 0,
        }
        for _, row in tqdm(test_set.iterrows(), "score_sample"):
            gt = row.ground_truth
            answer = row.answer
            context = row.context
            scores = self.score_sample(answer, gt, context)
            
            for metric, score_list in scores.items():
                score = score_list[0][0] if isinstance(score_list[0], list) else score_list[0]
                if score > threshold_mid:
                    if score > threshold_good:
                        good_models_count[metric] += 1
                    else:
                        mid_models_count[metric] += 1
                else:
                    bad_models_count[metric] += 1

            if not res:
                res = scores
            else:
                for k, v in scores.items():
                    res[k].extend(v)
        
        # Вычисляем средние значения
        for k, v in res.items():
            res[k] = np.mean(v)  # Меняем для корректного вычисления среднего
            
        total_samples = len(test_set)
        good_models_percentage = {
            k: (v / total_samples) * 100 for k, v in good_models_count.items()
        }
        mid_models_percentage = {
            i: (j / total_samples) * 100 for i, j in mid_models_count.items()
        }
        bad_models_percentage = {
            p: (l / total_samples) * 100 for p, l in bad_models_count.items()
        }

        return res, mid_models_percentage, good_models_percentage, bad_models_percentage

