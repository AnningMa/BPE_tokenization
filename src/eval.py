import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score

from bpe_fast import FastBPE
from bpe_naive import NaiveBPE
from morfessor_segmenter import MorfessorModel
from porter_segmenter_nltk import PorterSegmenter
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
            vocab_path = DEFAULT_VOCAB_DIR / f"{data_id}_v{vocab_size}_vocab.txt"
            merges_path = DEFAULT_VOCAB_DIR / f"{data_id}_v{vocab_size}_merges.txt"
            bpe = FastBPE()
            bpe.load_vocab(vocab_path, merges_path)
            if not is_long:
                self.tokenizer = bpe.tokenize
            else:
                self.tokenizer = bpe.tokenize_longest

        if type == "wpc":
            wp_voc = load_wp_vocab(
                DEFAULT_VOCAB_DIR
                / f"{data_id}_v{vocab_size}_m{min_pair_freq}_vocab.txt"
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

            def _f_wpc(word, vocab=wp_voc):
                res = WordPieceTrieTokenizer(vocab).encode_word_type(word)
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
    for w in corpus:
        pieces_a, pieces_b = tok_a(w), tok_b(w)
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


def make_vocab(
    base_name="Salesforce/wikitext",
    dataset_id="wikitext-103-v1",
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

    dataset = load_dataset(base_name, dataset_id, split=split)
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


def against_gold(gold_path, tokenize) -> Dict:

    gold = get_gold(gold_path)
    n_words = len(gold)
    n_sw_gold = sum([len(sw) for sw in gold.values()])
    avg_spw_gold = n_sw_gold / n_words

    vec_gold = []
    vec_pred = []
    n_sw_pred = []
    per_word_kappas = []
    per_word_f1s = []
    for word in gold.keys():
        v_gold = seg_to_vec(gold[word], len(word))
        vec_gold.extend(v_gold)

        pieces = tokenize(word)
        v_pred = seg_to_vec(pieces, len(word))
        vec_pred.extend(v_pred)
        n_sw_pred.append(len(pieces))

        if len(set(v_gold)) > 1 and len(set(v_pred)) > 1:
            per_word_kappas.append(cohen_kappa_score(v_pred, v_gold, labels=[0, 1]))
        else:
            per_word_kappas.append(np.nan)
        _, _, pw_f1 = my_p_r_f1(v_pred, v_gold)
        per_word_f1s.append(pw_f1)

    avg_spw_pred = sum(n_sw_pred) / n_words

    kappa = cohen_kappa_score(vec_pred, vec_gold)
    p, r, f1 = my_p_r_f1(vec_pred, vec_gold)

    return {
        "kappa": kappa,
        "precision": p,
        "recall": r,
        "f1": f1,
        "avg_spw_pred": avg_spw_pred,
        "avg_spw_gold": avg_spw_gold,
        "per_word_kappa": float(np.nanmean(per_word_kappas))
        if per_word_kappas
        else 0.0,
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

    preserved = set()
    n_subwords = []
    for w in freq_vocab:
        pieces = tokenize(w)
        n_subwords.append(len(pieces))
        if len(pieces) == 1:
            preserved.add(w)

    preserved_1k = set()
    for w in freq_vocab[:1000]:
        pieces = tokenize(w)
        if len(pieces) == 1:
            preserved_1k.add(w)

    n_pres_10k = len(preserved)
    # prop_10k = len(preserved) / len(freq_vocab)

    n_pres_1k = len(preserved_1k)
    # prop_1k = len(preserved_1k) / 1000

    avg_fertility = sum(n_subwords) / len(n_subwords)

    return {
        "avg_fertility": avg_fertility,
        "n_preserved(0k)": n_pres_1k,
        "n_preserved(10k)": n_pres_10k,
    }


def least_words_fert(tokenize) -> float:
    cpt = make_vocab()
    least_10k = cpt.most_common()[:-1_001:-1]

    n_sw = []
    for word in least_10k:
        n_sw.append(len(tokenize(word[0])))
    avg_fert = sum(n_sw) / len(n_sw)

    return avg_fert


def avg_fert_over_wt(tokenize, text):
    n_sw = []
    wt = re.split(r"\W+", text)
    for w in wt:
        n_sw.append(len(tokenize(w)))

    return sum(n_sw) / len(wt)


def in_out_domain(tokenize, domain):
    gtn = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    text_gtn = next(iter(gtn.skip(1_000_000)))["text"]
    while len(text_gtn.split()) <= 100:
        text_gtn = next(iter(gtn))["text"]

    wk = load_dataset(
        "Salesforce/wikitext", "wikitext-103-v1", split="test", streaming=True
    )
    text_wk = next(iter(wk.skip(10)))["text"]
    while len(text_wk.split()) <= 100:
        text_gtn = next(iter(wk))["text"]

    if domain == "wiki":
        in_domain_fert = avg_fert_over_wt(tokenize, text_wk)
        out_domain_fert = avg_fert_over_wt(tokenize, text_gtn)
    elif domain == "guten":
        in_domain_fert = avg_fert_over_wt(tokenize, text_gtn)
        out_domain_fert = avg_fert_over_wt(tokenize, text_wk)
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


def compare_run(tok_a: Tokenizer, tok_b: Tokenizer):
    print(f"\n---{tok_a.type} vs {tok_b.type}---")
    res = pairwise_agreement(agree_corpus, tok_a.tokenizer, tok_b.tokenizer)
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
    print(f"Evaluating {tok.type} ...")
    res_gold = against_gold(GOLD_PATH, tok.tokenizer)
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
        "morfessor": Tokenizer("morf"),
        "bpe_wiki_10000": Tokenizer("bpe", "wiki", 10_000),
        "bpe_wiki_10000_long": Tokenizer("bpe", "wiki", 10_000, is_long=True),
        "wpc_wiki_5000_500": Tokenizer("wpc", "wiki", 5_000, 500),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--which", "-w", choices=["compare", "eval"], required=True)
    parser.add_argument("--tok-a", "-a", choices=tokenizers.keys(), required=True)
    parser.add_argument("--tok-b", "-b", choices=tokenizers.keys())
    parser.add_argument("--eg", action="store_true")
    parser.add_argument("--test-data", action="store_true")
    args = parser.parse_args()

    os.makedirs(DEFAULT_VOCAB_DIR, exist_ok=True)

    # test_vocab = dict(make_vocab(split="test"))
    agree_corpus = get_gold(GOLD_PATH).keys()
    tok_a = tokenizers[args.tok_a]

    if args.which == "compare":
        if args.tok_b:
            tok_b = tokenizers[args.tok_b]
            compare_run(tok_a, tok_b)
        else:
            raise Exception("Two tokenizers needed")
    else:
        eval_run(tok_a)

    if args.eg:
        print(f"\n---Examples with {args.tok_a}")
        words = ["unbelievable", "tokenization", "preprocessing", "cats", "the"]
        for w in words:
            print(f"{w} -> {tok_a.tokenizer(w)}")


"""
SAVE_VOCAB_PATH = Path(__file__).parent.parent / "data"
if args.train:
    vocab = dict(make_vocab())
    if args.test_data:
        vocab = dict(itertools.islice(vocab.items(), 20_000))
    print("Training naive BPE tokenizer...")
    bpe = NaiveBPE()
    t0 = time.perf_counter()
    bpe.train(vocab_size=args.bpe_vocab, word_freqs=vocab)
    t1 = time.perf_counter()
    print(f"Naive BPE trained in {t1 - t0:.2f}s.")
    save_vocab(set(bpe.vocab), str(SAVE_VOCAB_PATH / "bpe-vocab.txt"))
    print(f"BPE vocabulary saved to {str(SAVE_VOCAB_PATH / 'bpe-vocab.txt')}")

    print("\nTraining fast BPE tokenizer...")
    f_bpe = FastBPE()
    t0 = time.perf_counter()
    f_bpe.train(vocab_size=args.bpe_vocab, word_freqs=vocab)
    t1 = time.perf_counter()
    print(f"Fast BPE trained in {t1 - t0:.2f}s.")

    print("\nTraining Word-Piece tokenizer...")
    t0 = time.perf_counter()
    wp_voc, _, _ = train_wordpiece(
        vocab,
        args.wp_vocab,
    )
    t1 = time.perf_counter()
    print(f"Word-Piece trained in {t1 - t0:.2f}s")
    save_vocab(wp_voc, str(SAVE_VOCAB_PATH / "wp-vocab.txt"))
    print(f"WordPiece vocabulary saved to: {str(SAVE_VOCAB_PATH / 'wp-vocab.txt')}")
"""
