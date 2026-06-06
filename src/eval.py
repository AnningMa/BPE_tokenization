import re
from collections import Counter, defaultdict
from typing import Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score

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
    write_output=False,
    output="../data/wikitext103_vocab.txt",
) -> Counter:
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


def against_gold(gold_path, tokenize):

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

    abs_pred = sum(vec_pred)
    abs_gold = sum(vec_gold)
    intersec = sum(a == 1 and b == 1 for a, b in zip(vec_pred, vec_gold))
    p = intersec / abs_pred if abs_pred > 0 else 0.0
    r = intersec / abs_gold if abs_gold > 0 else 0.0
    f1 = 2 * intersec / (abs_pred + abs_gold) if (abs_pred + abs_gold) > 0 else 0.0

    return {
        "kappa": kappa,
        "precision": p,
        "recall": r,
        "f1": f1,
        "avg_spw_pred": avg_spw_pred,
        "avg_spe_gold": avg_spw_gold,
    }


def freq_words_metrics(path, tokenize):
    freq_vocab = []
    with open(path) as f:
        for line in f:
            freq_vocab.append(line.strip())

    preserved = set()
    n_subwords = []
    for w in freq_vocab:
        pieces = tokenize(w)
        n_subwords.append(len(pieces))
        if len(pieces) == 1 and pieces[0] == w:
            preserved.add(w)

    preserved_5k = set()
    for w in freq_vocab[:5000]:
        pieces = tokenize(w)
        if len(pieces) == 1 and pieces[0] == w:
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


def least_words_fert(tokenize):
    ct = list(reversed(make_vocab()))
    least_10k_ct = ct[:10_000]
    least_10k = [e[0] for e in least_10k_ct]

    n_sw = []
    for word in least_10k:
        n_sw.append(len(tokenize(word)))
    avg_fert = sum(n_sw) / len(n_sw)

    return avg_fert


from morfessor_segmenter import MorfessorModel
from porter_segmenter_nltk import PorterSegmenter

porter_seg = PorterSegmenter()

mo = MorfessorModel()
mo.load("../data/morf_wiki_103.bin")

tokenizers = {
    "porter": porter_seg.segment,
    "morfessor": mo.segment,
}

print(against_gold(GOLD_PATH, tokenizers["morfessor"]))
print(freq_words_metrics(FREQ_WORDS_PATH, tokenizers["morfessor"]))
print(least_words_fert(tokenizers["morfessor"]))
