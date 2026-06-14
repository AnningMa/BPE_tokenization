import argparse
import itertools
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score

# from bpe_fast import FastBPE
from bpe_naive import NaiveBPE
from morfessor_segmenter import MorfessorModel
from porter_segmenter_nltk import PorterSegmenter
from wordpiece_baseline import encode_word_type, save_vocab, train_wordpiece
from wordpiece_fast_tokenization import WordPieceTrieTokenizer

GOLD_PATH = Path(__file__).parent.parent / "data" / "goldstd_combined.segmentation.eng"
# "../data/goldstd_combined.segmentation.eng"
FREQ_WORDS_PATH = Path(__file__).parent.parent / "data" / "google-10000-english.txt"
# "../data/google-10000-english.txt"


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
        "n_preserved(10k)": n_pres_10k,
        # "proportion(10k)": prop_10k,
        "n_preserved(1k)": n_pres_1k,
        # "proportion(1k)": prop_1k,
    }


def least_words_fert(tokenize) -> float:
    cpt = make_vocab()
    least_10k = cpt.most_common()[:-1_001:-1]

    n_sw = []
    for word in least_10k:
        n_sw.append(len(tokenize(word[0])))
    avg_fert = sum(n_sw) / len(n_sw)

    return avg_fert


def compare_run(tok_a: str, tok_b: str):
    print(f"\n---{tok_a} vs {tok_b}---")
    res = pairwise_agreement(agree_corpus, tokenizers[tok_a], tokenizers[tok_b])
    print(f"F1: {res['f1']:.3f}\nAverage per word F1: {res['per_word_f1']:.3f}")


def eval_run(tok: str):
    print(f"\n---{tok}---")
    res_gold = against_gold(GOLD_PATH, tokenizers[tok])
    res_freq = freq_words_metrics(FREQ_WORDS_PATH, tokenizers[tok])
    res_rare = least_words_fert(tokenizers[tok])
    print("Against gold set:")
    print(f"""Precision: {res_gold["precision"]:.3f};
Recall: {res_gold["recall"]:.3f};
F1 score:  {res_gold["f1"]:.3f};
Average F1 per word: {res_gold["per_word_f1"]:.3f};
Average number of subwords per word (gold set): {res_gold["avg_spw_gold"]:.3f};
Average number of subwords per word (tokenizer): {res_gold["avg_spw_pred"]:.3f};
""")
    print("On frequent words:")
    print(f"""Average fertility: {res_freq["avg_fertility"]:.3f};
Number of preserved word types (among top 10k): {res_freq["n_preserved(10k)"]};
Number of preserved word types (among top 1k): {res_freq["n_preserved(1k)"]};
""")
    print("On rare words:")
    print(
        f"Average fertility (among the most rare 1k of the training corpus): {res_rare:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-eg", action="store_true")
    parser.add_argument("--test-data", action="store_true")
    parser.add_argument("--bpe-vocab", type=int, default=5_000)
    parser.add_argument("--wp-vocab", type=int, default=10_000)
    args = parser.parse_args()

    SAVE_VOCAB_PATH = Path(__file__).parent.parent / "data"

    porter_seg = PorterSegmenter()
    mo = MorfessorModel()
    mo.load("../data/morf_wiki_103.bin")
    vocab = dict(make_vocab())

    if args.test_data:
        vocab = dict(itertools.islice(vocab.items(), 20_000))

    """
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
    """

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

    def wpc(word, vocab=wp_voc):
        res = encode_word_type(word, vocab)
        return [w.replace("##", "") for w in res]

    def f_wpc(word, vocab=wp_voc):
        res = WordPieceTrieTokenizer(vocab).encode_word_type(word)
        return [w.replace("##", "") for w in res]

    tokenizers = {
        "porter": porter_seg.segment,
        "morfessor": mo.segment,
        # "bpe": bpe.tokenize,
        # "bpe_long": bpe.tokenize_longest,
        # "fast_bpe": f_bpe.tokenize,
        "wpc": wpc,
        "fast_wpc": f_wpc,
    }

    test_vocab = dict(make_vocab(split="test"))
    agree_corpus = test_vocab.keys()

    """
    4 个指标：

    1. pairwise agreement：输入1个测试集（单词表，我这里暂时用了wiki103的test split，可以换），
    2个tokenizer方法，输出2个方法的整个测试集上boundary位置的kappa和f1；

    2. against gold：输入gold测试集+1个tokenizer方法，输出和gold对比的kappa，precision，recall，f1，
    gold的每词平均子词（subword）数，tokenizer预测的每词平均子词数；

    3. freq words metrics：对于英语中前10000频繁的词（来源：https://github.com/first20hours/google-10000-english）
    输入词表路径和1个tokenizer方法，输出这个tokenizer在前10000/1000词中保留（即没做任何切分）的数量和比例，
    也输出前10000词平均fertility（一个词分出来几个子词）

    4. least words fert：输入一个tokenizer方法，输出它在训练集中最罕见的10000词上的平均fertility
    """

    ONLY_EXAMPLES = False

    if not args.only_eg:
        print("\n---Agreement---")
        # compare_run("bpe", "wpc")
        # compare_run("bpe_long", "wpc")
        # compare_run("bpe", "morfessor")
        compare_run("wpc", "morfessor")

        eval_run("porter")
        eval_run("morfessor")
        # eval_run("bpe")
        # eval_run("bpe_long")
        eval_run("wpc")
        # eval_run("fast_bpe")
        eval_run("fast_wpc")

    print("\n---Examples---")
    words = ["unbelievable", "tokenization", "preprocessing", "cats", "the"]
    for tok in tokenizers.keys():
        print(tok)
        for w in words:
            print(f"{w} -> {tokenizers[tok](w)}")
