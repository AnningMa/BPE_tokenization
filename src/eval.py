import re
from collections import Counter, defaultdict
from typing import Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score

GOLD_PATH = "../gold/goldstd_combined.segmentation.eng"


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


def pairwise_agreement(corpus_words, tokenizers):
    all_decisions = defaultdict(list)

    for word, freq in corpus_words:
        if len(word) < 2:
            continue

        for name, tokenize in tokenizers.items():
            pieces = tokenize(word)
            vec = seg_to_vec(pieces, len(word))
            all_decisions[name].extend(vec * freq)  # weight by freq

    names = list(tokenizers.keys())
    res = {}

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            vec_a = all_decisions[a]
            vec_b = all_decisions[b]

            kappa = cohen_kappa_score(vec_a, vec_b)

            a_abs = sum(vec_a)
            b_abs = sum(vec_b)
            intersec = sum(a == 1 and b == 1 for a, b in zip(vec_a, vec_b))
            f1 = 2 * intersec / (a_abs + b_abs) if (a_abs + b_abs) > 0 else 0.0

            res[(a, b)] = {"kappa": kappa, "f1": f1}

    return res


def make_vocab(
    base_name="Salesforce/wikitext",
    dataset_id="wikitext-103-v1",
    output="../data/wikitext103_vocab.txt",
) -> Counter:
    dataset = load_dataset(base_name, dataset_id, split="train")
    counter = Counter()

    for e in dataset:
        text = e["text"].lower()
        words = re.findall(r"\w+\S*", text)
        counter.update(words)

    with open(output, "w") as f:
        for word, count in counter.items():
            f.write(f"{count} {word}\n")

    print(f"Vocabulary size: {len(counter)}")
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


def against_gold(gold_path, tokenizers):
    all_decisions = defaultdict(list)

    gold: Dict[str, list] = get_gold(gold_path)
    for word in gold.keys():
        vec = seg_to_vec(gold[word], len(word))
        all_decisions["gold"].extend(vec)

        for name, tokenize in tokenizers.items():
            pieces = tokenize(word)
            vec = seg_to_vec(pieces, len(word))
            all_decisions[name].extend(vec)

    names = list(tokenizers.keys())
    res = {}
    vec_g = all_decisions["gold"]
    g_abs = sum(vec_g)
    for i, n in enumerate(names):
        vec_n = all_decisions[n]

        kappa = cohen_kappa_score(vec_n, vec_g)

        n_abs = sum(vec_n)
        intersec = sum(a == 1 and b == 1 for a, b in zip(vec_n, vec_g))
        precision = intersec / n_abs if n_abs > 0 else 0.0
        recall = intersec / g_abs if g_abs > 0 else 0.0
        f1 = 2 * intersec / (n_abs + g_abs) if (n_abs + g_abs) > 0 else 0.0

        res[n] = {"kappa": kappa, "precision": precision, "recall": recall, "f1": f1}

    return res


from morfessor_segmenter import MorfessorModel
from porter_segmenter_nltk import PorterSegmenter

porter_seg = PorterSegmenter()

mo = MorfessorModel()
mo.load("../data/morf_wiki_103.bin")

tokenizers = {
    "porter": porter_seg.segment,
    "morfessor": mo.segment,
}

print(against_gold(GOLD_PATH, tokenizers))
