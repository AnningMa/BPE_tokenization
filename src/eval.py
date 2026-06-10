import re
from collections import Counter, defaultdict
from typing import Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score

from morfessor_segmenter import MorfessorModel
from porter_segmenter_nltk import PorterSegmenter

GOLD_PATH = "../data/goldstd_combined.segmentation.eng"
FREQ_WORDS_PATH = "../data/google-10000-english.txt"


def seg_to_vec(pieces: list[str], word_len: int) -> list[int]:
    boundaries = []
    i = 0
    for piece in pieces[:-1]:  # last piece has no boundary after it
        i += len(piece)
        boundaries.append(i)

    vec = [0] * (word_len - 1)
    for b in boundaries:
        if 0 < b < word_len:
            vec[b - 1] = 1
    return vec


def my_p_r_f1(x, y):
    x_abs = sum(x)
    y_abs = sum(y)
    inter = sum(a == 1 and b == 1 for a, b in zip(x, y))
    p = inter / x_abs if x_abs > 0 else 0.0
    r = inter / y_abs if y_abs > 0 else 0.0
    f1 = 2 * inter / (x_abs + y_abs) if (x_abs + y_abs) > 0 else 0.0
    return p, r, f1


def pairwise_agreement(corpus, tok_a, tok_b) -> Dict[str, float]:
    vec_a, vec_b = [], []
    for w in corpus:
        pieces_a, pieces_b = tok_a(w), tok_b(w)
        v_a = seg_to_vec(pieces_a, len(w))
        v_b = seg_to_vec(pieces_b, len(w))
        vec_a.extend(v_a)
        vec_b.extend(v_b)

    kappa = cohen_kappa_score(vec_a, vec_b)
    _, _, f1 = my_p_r_f1(vec_a, vec_b)
    return {"cohen's kappa": kappa, "f1": f1}


_vocab_cache: Counter | None = None


def make_vocab(
    base_name="Salesforce/wikitext",
    dataset_id="wikitext-103-v1",
    write_output=False,
    output="../data/wikitext103_vocab.txt",
) -> Counter:
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache

    dataset = load_dataset(base_name, dataset_id, split="train")
    counter = Counter()

    for e in dataset:
        text = e["text"].lower()
        words = re.findall(r"\w+\S*", text)
        counter.update(words)

    if write_output:
        with open(output, "w") as f:
            for word, count in counter.items():
                f.write(f"{count} {word}\n")

    print(f"Vocabulary size: {len(counter)}")
    _vocab_cache = counter
    return counter


def get_gold(input_path) -> Dict[str, list]:
    res = {}
    with open(input_path) as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            word, analyses = line.split("\t")
            if "," in analyses:
                analyses = analyses.split(",")[0]
            morphemes = analyses.split()
            surfaces = []
            for m in morphemes:
                surface = m.split(":")[0]
                if (
                    not surface.startswith("+")
                    and not surface.startswith("~")
                    and surface
                ):
                    surfaces.append(surface)
            if surfaces:
                res[word] = surfaces
    return res


def against_gold(gold_path, tokenize) -> Dict[str, float]:

    gold = get_gold(gold_path)
    n_words = len(gold)
    n_sw_gold = sum([len(sw) for sw in gold.values()])
    avg_spw_gold = n_sw_gold / n_words

    vec_gold = []
    vec_pred = []
    n_sw_pred = []
    for word in gold.keys():
        vec_gold.extend(seg_to_vec(gold[word], len(word)))

        pieces = tokenize(word)
        vec_pred.extend(seg_to_vec(pieces, len(word)))
        n_sw_pred.append(len(pieces))
    avg_spw_pred = sum(n_sw_pred) / n_words

    kappa = cohen_kappa_score(vec_pred, vec_gold)
    p, r, f1 = my_p_r_f1(vec_pred, vec_gold)

    return {
        "kappa": kappa,
        "precision": p,
        "recall": r,
        "f1": f1,
        "avg_spw_pred": avg_spw_pred,
        "avg_spe_gold": avg_spw_gold,
    }


def freq_words_metrics(path, tokenize) -> Dict[str, float]:

    freq_vocab = []
    with open(path) as f:
        for line in f:
            freq_vocab.append(line.strip())

    preserved = set()
    n_subwords = []
    for w in freq_vocab:
        pieces = tokenize(w)
        n_subwords.append(len(pieces))
        if len(pieces) == 1:
            preserved.add(w)

    preserved_5k = set()
    for w in freq_vocab[:5000]:
        pieces = tokenize(w)
        if len(pieces) == 1:
            preserved_5k.add(w)

    n_pres_10k = len(preserved)
    prop_10k = len(preserved) / len(freq_vocab)

    n_pres_5k = len(preserved_5k)
    prop_5k = len(preserved_5k) / 5000

    avg_fertility = sum(n_subwords) / len(n_subwords)

    return {
        "avg_fertility": avg_fertility,
        "n_preserved(10k)": n_pres_10k,
        "proportion(10k)": prop_10k,
        "n_preserved(5k)": n_pres_5k,
        "proportion(5k)": prop_5k,
    }


def least_words_fert(tokenize) -> float:
    cpt = make_vocab()
    least_10k = cpt.most_common()[:-10_001:-1]

    n_sw = []
    for word in least_10k:
        n_sw.append(len(tokenize(word)))
    avg_fert = sum(n_sw) / len(n_sw)

    return avg_fert


if __name__ == "__main__":
    porter_seg = PorterSegmenter()

    mo = MorfessorModel()
    mo.load("../data/morf_wiki_103.bin")

    tokenizers = {
        "porter": porter_seg.segment,
        "morfessor": mo.segment,
    }

    gold = get_gold(GOLD_PATH)

    """
    4 个指标：

    1. pairwise agreement：输入1个测试集（单词表，我这里暂时用了gold，可以换），
    2个tokenizer方法，输出2个方法的整个测试集上boundary位置的kappa和f1；

    2. against gold：输入gold测试集+1个tokenizer方法，输出和gold对比的kappa，precision，recall，f1，
    gold的每词平均子词（subword）数，tokenizer预测的每词平均子词数；

    3. freq words metrics：对于英语中前10000频繁的词（来源：https://github.com/first20hours/google-10000-english）
    输入词表路径和1个tokenizer方法，输出这个tokenizer在前10000/5000词中保留（即没做任何切分）的数量和比例，
    也输出前10000词平均fertility（一个词分出来几个子词）

    4. least words fert：输入一个tokenizer方法，输出它在训练集中最罕见的10000词上的平均fertility
    """

    print(pairwise_agreement(gold, tokenizers["porter"], tokenizers["morfessor"]))
    print(against_gold(GOLD_PATH, tokenizers["morfessor"]))
    print(freq_words_metrics(FREQ_WORDS_PATH, tokenizers["morfessor"]))
    print(least_words_fert(tokenizers["morfessor"]))
