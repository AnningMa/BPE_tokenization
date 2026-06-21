import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from tarfile import DEFAULT_FORMAT
from typing import Dict, List

import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn.metrics import cohen_kappa_score

from bpe_fast import FastBPE
from bpe_naive import NaiveBPE
from morfessor_segmenter import MorfessorModel
from porter_segmenter_nltk import PorterSegmenter
from src.get_ref_text import text_gtn
from wordpiece_baseline import encode_word_type, save_vocab, train_wordpiece
from wordpiece_fast_tokenization import WordPieceTrieTokenizer

"""
three metrics

1. BPE-WordPiece agreement:
    - boundary F1 on corpus
    - average boundary F1 per word.

2. Against gold standard:
    - precision, recall, F1
    - average fertility (nb. of subwords per word token)

3. On freq/rare words:
    - nb. of untouched freq words
    - avg fertility (on both freq and rare words)
"""


GOLD_PATH = Path(__file__).parent.parent / "data" / "goldstd_combined.segmentation.eng"
# "../data/goldstd_combined.segmentation.eng"
FREQ_WORDS_PATH = Path(__file__).parent.parent / "data" / "google-10000-english.txt"
# "../data/google-10000-english.txt"
LOG_PATH = Path(__file__).parent.parent / "log"

DEFAULT_VOCAB_DIR = Path(__file__).parent.parent / "data"


class Tokenizer:
    def __init__(
        self,
        type: str,
        data_id: str | None = None,
        vocab_size: int | None = None,
        min_pair_freq: int | None = None,
        is_long: bool = False,
    ):
        self.type = type
        self.vocab_size = vocab_size
        self.min_pair_freq = min_pair_freq
        self.data_id = data_id

        if type == "bpe":
            vocab_path = DEFAULT_VOCAB_DIR / f"bpe_{data_id}_v{vocab_size}_vocab.txt"
            merges_path = DEFAULT_VOCAB_DIR / f"bpe_{data_id}_v{vocab_size}_merges.txt"
            bpe = FastBPE()
            bpe.load_vocab(vocab_path, merges_path)
            if not is_long:
                self.tokenizer = bpe.tokenize
            else:
                self.tokenizer = bpe.tokenize_longest

        if type == "wpc":
            wp_voc = load_wp_vocab(
                DEFAULT_VOCAB_DIR
                / f"wpc_{data_id}_v{vocab_size}_m{min_pair_freq}_vocab.txt"
            )

            def _wpc(word, vocab=wp_voc):
                res = encode_word_type(word, vocab)
                return [w.replace("##", "") for w in res]

            self.tokenizer = _wpc

        if type == "f_wpc":
            wp_voc = load_wp_vocab(
                DEFAULT_VOCAB_DIR
                / f"{data_id}_v{vocab_size}_m{min_pair_freq}_vocab.txt"
            )
            wptk = WordPieceTrieTokenizer(wp_voc)

            def _f_wpc(word):
                res = wptk.encode_word_type(word)
                return [w.replace("##", "") for w in res]

            self.tokenizer = _f_wpc

        if type == "morf":
            mo = MorfessorModel().load("../data/morf_wiki_103.bin")
            self.tokenizer = mo.segment


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


def pairwise_agreement(corpus, tok_a, tok_b) -> Dict:
    vec_a, vec_b = [], []
    per_word_kappas = []
    per_word_f1s = []
    skipped = 0
    for w in corpus:
        try:
            pieces_a, pieces_b = tok_a(w), tok_b(w)
        except KeyError:
            skipped += 1
            continue
        v_a = seg_to_vec(pieces_a, len(w))
        v_b = seg_to_vec(pieces_b, len(w))
        vec_a.extend(v_a)
        vec_b.extend(v_b)

        if len(set(v_a)) > 1 and len(set(v_b)) > 1:
            kw = cohen_kappa_score(v_a, v_b, labels=[0, 1])
        else:
            kw = 1.0 if v_a == v_b else 0.0

        per_word_kappas.append(kw)
        _, _, pw_f1 = my_p_r_f1(v_a, v_b)
        per_word_f1s.append(pw_f1)

    if skipped:
        print(f"[WARN] Skipped {skipped} OOV words in pairwise agreement")

    kappa = cohen_kappa_score(vec_a, vec_b)
    _, _, f1 = my_p_r_f1(vec_a, vec_b)
    return {
        "kappa": kappa,
        "f1": f1,
        "per_word_kappa": float(np.nanmean(per_word_kappas))
        if per_word_kappas
        else 0.0,
        "per_word_f1": sum(per_word_f1s) / len(per_word_f1s) if per_word_f1s else 0.0,
    }


_train_vocab_cache: Counter | None = None
_test_vocab_cache: Counter | None = None
LOCAL_DIR = Path(__file__).parent.parent / "data"


def make_vocab(
    base_name="Salesforce/wikitext",
    dataset_id="wikitext-103-v1",
    local_dir=LOCAL_DIR / "wikitext103",
    write_output=False,
    output="../data/wikitext103_vocab.txt",
    split="train",
) -> Counter:
    if split == "train":
        global _train_vocab_cache
        if _train_vocab_cache is not None:
            return _train_vocab_cache
    elif split == "test":
        global _test_vocab_cache
        if _test_vocab_cache is not None:
            return _test_vocab_cache

    try:
        dataset = load_from_disk(local_dir)

    except Exception:
        print("No local data, start downloading...")
        dataset = load_dataset(base_name, dataset_id, split=split, token=HF_TOKEN)

    counter = Counter()

    for e in dataset:
        text = e["text"].strip().lower()
        words = re.findall(r"\b[a-z'-]+\b", text)
        counter.update(words)

    if write_output:
        with open(output, "w") as f:
            for word, count in counter.items():
                f.write(f"{count} {word}\n")

    print(f"{split} vocabulary size: {len(counter)}")

    if split == "train":
        _train_vocab_cache = counter
    elif split == "test":
        _test_vocab_cache = counter

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


def against_gold(gold, tokenize) -> Dict:

    n_words = len(gold)
    n_sw_gold = sum([len(sw) for sw in gold.values()])
    avg_spw_gold = n_sw_gold / n_words

    vec_gold = []
    vec_pred = []
    n_sw_pred = []
    # per_word_kappas = []
    per_word_f1s = []
    skipped = 0
    for word in gold.keys():
        v_gold = seg_to_vec(gold[word], len(word))
        vec_gold.extend(v_gold)

        try:
            pieces = tokenize(word)
        except KeyError:
            skipped += 1
            pieces = [word]

        v_pred = seg_to_vec(pieces, len(word))
        vec_pred.extend(v_pred)
        n_sw_pred.append(len(pieces))

        """
        if len(set(v_gold)) > 1 and len(set(v_pred)) > 1:
            per_word_kappas.append(cohen_kappa_score(v_pred, v_gold, labels=[0, 1]))
        else:
            per_word_kappas.append(np.nan)
        """
        _, _, pw_f1 = my_p_r_f1(v_pred, v_gold)
        per_word_f1s.append(pw_f1)
    print(f"[WARN] Skipped {skipped} OOV words in against_gold")

    avg_spw_pred = sum(n_sw_pred) / n_words

    # kappa = cohen_kappa_score(vec_pred, vec_gold)
    p, r, f1 = my_p_r_f1(vec_pred, vec_gold)

    return {
        # "kappa": kappa,
        "precision": p,
        "recall": r,
        "f1": f1,
        "avg_spw_pred": avg_spw_pred,
        "avg_spw_gold": avg_spw_gold,
        # "per_word_kappa": float(np.nanmean(per_word_kappas)) if per_word_kappas else 0.0,
        "per_word_f1": sum(per_word_f1s) / len(per_word_f1s) if per_word_f1s else 0.0,
    }


_freq_vocab: List | None = None


def get_freq_vocab(path):
    global _freq_vocab
    if _freq_vocab is not None:
        return _freq_vocab
    else:
        freq_vocab = []
        with open(path) as f:
            for line in f:
                freq_vocab.append(line.strip())
        _freq_vocab = freq_vocab
        return freq_vocab


def freq_words_metrics(path, tokenize) -> Dict:
    freq_vocab = get_freq_vocab(path)

    preserved_1k = set()
    preserved_10k = set()
    n_subwords = []
    n_subwords_1k = []
    skipped = 0
    for i, w in enumerate(freq_vocab):
        try:
            pieces = tokenize(w)
        except KeyError:
            skipped += 1
            continue

        n = len(pieces)
        n_subwords.append(n)
        if n == 1:
            preserved_10k.add(w)
            if i < 1000:
                preserved_1k.add(w)
        if i < 1000:
            n_subwords_1k.append(n)
    print(f"[WARN] Skipped {skipped} OOV words in freq_words_metrics")

    return {
        "avg_fertility(1k)": sum(n_subwords_1k) / len(n_subwords_1k),
        "avg_fertility(10k)": sum(n_subwords) / len(n_subwords),
        "n_preserved(1k)": len(preserved_1k),
        "n_preserved(10k)": len(preserved_10k),
    }


_least_1k_cache = None


def get_least_1k():
    global _least_1k_cache
    if _least_1k_cache is None:
        cpt = make_vocab()
        _least_1k_cache = cpt.most_common()[:-1001:-1]
    return _least_1k_cache


def least_words_fert(tokenize) -> float:
    least_1k = get_least_1k()

    n_sw = []
    skipped = 0
    for word in least_1k:
        try:
            n_sw.append(len(tokenize(word[0])))
        except KeyError:
            skipped += 1
            continue

    print(f"[WARN] Skipped {skipped} OOV words in least_words_fert")
    avg_fert = sum(n_sw) / len(n_sw)

    return avg_fert


_split_re = re.compile(r"\W+")


def avg_fert_over_wt(tokenize, lines):
    word_cache = {}
    total_sw = total_wt = 0
    for line in lines:
        wt = [w for w in re.split(r"\W+", line) if w]
        total_wt += len(wt)
        for w in wt:
            if w not in word_cache:
                word_cache[w] = len(tokenize(w))
            total_sw += word_cache[w]

    return total_sw / total_wt


_wiki_test_cache = None
_guten_test_cache = None


def load_lines(path):
    with open(path, "r") as f:
        return [line.rstrip() for line in f]


def in_out_domain(tokenize, domain):
    global _wiki_test_cache
    global _guten_test_cache

    if _wiki_test_cache is None:
        _wiki_test_cache = load_lines(DEFAULT_VOCAB_DIR / "wiki_test.txt")
    if _guten_test_cache is None:
        _guten_test_cache = load_lines(DEFAULT_VOCAB_DIR / "guten_test_chunk.txt")

    if domain.startswith("wiki"):
        in_domain_fert = avg_fert_over_wt(tokenize, _wiki_test_cache)
        out_domain_fert = avg_fert_over_wt(tokenize, _guten_test_cache)
    elif domain.startswith("guten"):
        in_domain_fert = avg_fert_over_wt(tokenize, _guten_test_cache)
        out_domain_fert = avg_fert_over_wt(tokenize, _wiki_test_cache)
    else:
        raise ValueError(f"Unkown doamin: {domain}")

    return {"in_domain": in_domain_fert, "out_domain": out_domain_fert}


def load_wp_vocab(path):
    wp_voc = set()
    with open(path, "r") as f:
        for w in f:
            wp_voc.add(w.strip())
    return wp_voc


def wpc(word, vocab):
    res = encode_word_type(word, vocab)
    return [w.replace("##", "") for w in res]


def f_wpc(word, vocab):
    res = WordPieceTrieTokenizer(vocab).encode_word_type(word)
    return [w.replace("##", "") for w in res]


def compare_run(tok_a: Tokenizer, tok_b: Tokenizer, corpus):
    print(f"\n---{tok_a.type} vs {tok_b.type}---")
    res = pairwise_agreement(corpus, tok_a.tokenizer, tok_b.tokenizer)
    params_a = {
        "type": tok_a.type,
        "train_data": tok_a.data_id,
        "vocab_size": tok_a.vocab_size,
        "min_pair_freq": tok_a.min_pair_freq,
    }
    params_b = {
        "type": tok_b.type,
        "train_data": tok_b.data_id,
        "vocab_size": tok_b.vocab_size,
        "min_pair_freq": tok_b.min_pair_freq,
    }
    with open(LOG_PATH / "agreement-log.jsonl", "a") as f:
        json.dump({"a": params_a, "b": params_b, "result": res}, f)
        f.write("\n")
    print(f"Log written to: {LOG_PATH / 'agreement-log.jsonl'}")


def eval_run(tok: Tokenizer):
    GOLD_CACHE = get_gold(GOLD_PATH)
    print(f"Evaluating {tok.type} ...")
    res_gold = against_gold(GOLD_CACHE, tok.tokenizer)
    res_freq = freq_words_metrics(FREQ_WORDS_PATH, tok.tokenizer)
    res_rare = least_words_fert(tok.tokenizer)
    res_domain = None
    if not tok.type == "morf":
        res_domain = in_out_domain(tok.tokenizer, tok.data_id)

    with open(LOG_PATH / "tokenize-log.jsonl", "a") as f:
        json.dump(
            {
                "type": tok.type,
                "train_data": tok.data_id,
                "vocab_size": tok.vocab_size,
                "min_pair_freq": tok.min_pair_freq,
                "against_gold": res_gold,
                "on_freq_words": res_freq,
                "on_rare_words": res_rare,
                "avg_fert_per_wt": res_domain,
            },
            f,
        )
        f.write("\n")
    print(f"log written to: {LOG_PATH / 'tokenize-log.jsonl'}")


if __name__ == "__main__":
    tokenizers = {
        "morfessor": Tokenizer("morf", "wiki"),
        "bpe_wiki_5000": Tokenizer("bpe", "wiki", 5_000),
        "bpe_wiki_10000": Tokenizer("bpe", "wiki", 10_000),
        "bpe_wiki_10000_long": Tokenizer("bpe", "wiki", 10_000, is_long=True),
        "bpe_wiki_20000": Tokenizer("bpe", "wiki", 20_000),
        "bpe_guten600_10000": Tokenizer("bpe", "guten600", 10_000),
        "bpe_guten1k2_10000": Tokenizer("bpe", "guten1k2", 10_000),
        "bpe_guten1k2_10000_long": Tokenizer("bpe", "guten1k2", 10_000, is_long=True),
        "wpc_wiki_10000_100": Tokenizer("wpc", "wiki", 10_000, 100),
        "wpc_wiki_10000_500": Tokenizer("wpc", "wiki", 10_000, 500),
        "wpc_wiki_20000_500": Tokenizer("wpc", "wiki", 20_000, 500),
        "wpc_guten600_10000_500": Tokenizer("wpc", "guten600", 10_000, 500),
        "wpc_guten1k2_10000_500": Tokenizer("wpc", "guten1k2", 10_000, 500),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--which", "-w", choices=["compare", "eval"], required=True)
    parser.add_argument("--tok-a", "-a", choices=tokenizers.keys(), required=True)
    parser.add_argument("--tok-b", "-b", choices=tokenizers.keys())
    parser.add_argument("--eg", action="store_true")
    parser.add_argument("--test-data", action="store_true")
    args = parser.parse_args()

    os.makedirs(DEFAULT_VOCAB_DIR, exist_ok=True)

    test_vocab = dict(
        make_vocab(local_dir=LOCAL_DIR / "wikitext103_test", split="test")
    )
    # agree_corpus = get_gold(GOLD_PATH).keys()
    tok_a = tokenizers[args.tok_a]

    if args.which == "compare":
        if args.tok_b:
            tok_b = tokenizers[args.tok_b]
            compare_run(tok_a, tok_b, test_vocab)
        else:
            raise Exception("Two tokenizers needed")
    else:
        eval_run(tok_a)

    if args.eg:
        print(f"\n---Examples with {args.tok_a}")
        words = ["unbelievable", "tokenization", "preprocessing", "cats", "the"]
        for w in words:
            print(f"{w} -> {tok_a.tokenizer(w)}")
