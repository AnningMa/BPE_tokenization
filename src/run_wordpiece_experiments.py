"""
Run WordPiece baseline and fast-version experiments.

This runner keeps the essential experiment logic:
1. train the baseline WordPiece vocabulary on the training split;
2. compare baseline tokenization with trie-based fast tokenization on the same eval word tokens;
3. optionally compare baseline vocabulary learning with experimental fast vocabulary learning
   on the same limited word-type frequency dictionary;
4. export WordPiece segmentations without ## for the evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections.abc import Iterable, Sequence


# Make imports work when running from the project root with: python src/run_wordpiece_experiments.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_corpus, pre_tokenize, get_word_type_frequencies

from wordpiece_baseline import ( 
    encode_word_type,
    train_wordpiece,
)
from wordpiece_fast_tokenization import WordPieceTrieTokenizer  
from wordpiece_fast_vocab_learning import time_baseline_and_fast_training


def collect_word_tokens(texts: Iterable[str], max_tokens: int) -> list[str]:
    """Pre-tokenize texts and keep the first max_tokens word tokens.

    If max_tokens <= 0, all word tokens are kept.
    """
    word_tokens: list[str] = []
    limit = None if max_tokens <= 0 else max_tokens

    for text in texts:
        word_tokens.extend(pre_tokenize(text))
        if limit is not None and len(word_tokens) >= limit:
            return word_tokens[:limit]

    return word_tokens


def limit_word_type_freqs(word_type_freqs: dict[str, int], max_word_types: int) -> dict[str, int]:
    """
    Keep the most frequent word types.
    This is used only for the experimental fast-training comparison. The same
    limited dictionary is passed to BOTH the baseline trainer and the fast trainer.
    """
    if max_word_types <= 0:
        return word_type_freqs

    return dict(
        sorted(word_type_freqs.items(), key=lambda item: item[1], reverse=True)[:max_word_types]
    )


def evaluate_tokenizer_outputs(outputs: Sequence[Sequence[str]]) -> dict[str, float]:
    """Compute simple tokenization metrics from already-tokenized word tokens."""
    n_words = len(outputs)
    n_subwords = sum(len(tokens) for tokens in outputs)
    n_unk = sum(1 for tokens in outputs if list(tokens) == ["[UNK]"])
    n_single = sum(1 for tokens in outputs if len(tokens) == 1 and list(tokens) != ["[UNK]"])

    return {
        "number_of_word_tokens": n_words,
        "number_of_subword_tokens": n_subwords,
        "average_subword_tokens_per_word_token": n_subwords / n_words if n_words else 0.0,
        "word_token_unk_rate": n_unk / n_words if n_words else 0.0,
        "word_token_single_subword_rate": n_single / n_words if n_words else 0.0,
    }


def time_tokenization_comparison(
    word_tokens: Sequence[str],
    vocabulary: set,
) -> tuple[dict[str, float], list[list[str]], list[list[str]], bool]:
    """Compare baseline and fast tokenization on the exact same word-token list."""
    fast_tokenizer = WordPieceTrieTokenizer(vocabulary)

    start = time.perf_counter()
    baseline_outputs = [encode_word_type(word, vocabulary) for word in word_tokens]
    baseline_time = time.perf_counter() - start

    start = time.perf_counter()
    fast_outputs = [fast_tokenizer.encode_word_type(word) for word in word_tokens]
    fast_time = time.perf_counter() - start

    same_outputs = baseline_outputs == fast_outputs
    timing = {
        "baseline_tokenization_time_seconds": baseline_time,
        "fast_tokenization_time_seconds": fast_time,
        "speedup": baseline_time / fast_time if fast_time > 0 else None,
    }
    return timing, baseline_outputs, fast_outputs, same_outputs


def strip_wordpiece_prefix(tokens: Sequence[str]) -> list[str]:
    """Remove WordPiece continuation markers for BPE/morphology evaluation."""
    pieces: list[str] = []
    for token in tokens:
        if token == "[UNK]":
            pieces.append(token)
        elif token.startswith("##"):
            pieces.append(token[2:])
        else:
            pieces.append(token)
    return pieces


def export_surface_segmentations(
    word_types: Sequence[str],
    vocabulary: set,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Save word -> surface pieces, with WordPiece ## markers removed."""
    segmentations = {
        word: strip_wordpiece_prefix(encode_word_type(word, vocabulary)) for word in word_types
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(segmentations, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word_type", "surface_pieces_for_evaluation"])
        for word, pieces in segmentations.items():
            writer.writerow([word, " ".join(pieces)])

    print("Example surface segmentations exported for evaluation:")
    for word, pieces in list(segmentations.items())[:10]:
        print(f"{word!r:<20} -> {pieces}")


def save_vocab(vocabulary: set, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for token in sorted(vocabulary):
            f.write(token + "\n")


def save_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.tokens")
    parser.add_argument("--valid_path", type=str, default="data/valid.tokens")
    parser.add_argument("--test_path", type=str, default="data/test.tokens")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_train_lines", type=int, default=None)
    parser.add_argument("--max_eval_word_tokens", type=int, default=100000)
    parser.add_argument("--max_fast_training_word_types", type=int, default=0)
    parser.add_argument("--verbose_merges", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load train split and learn baseline vocabulary.
    train_corpus = load_corpus(args.train_path)
    if args.max_train_lines is not None:
        train_corpus = train_corpus[: args.max_train_lines]

    word_type_freqs = get_word_type_frequencies(train_corpus)
    print("Number of training lines:", len(train_corpus))
    print("Number of training word types:", len(word_type_freqs))

    print("\n1) Baseline WordPiece vocabulary learning")
    start = time.perf_counter()
    baseline_vocabulary, _, baseline_merges = train_wordpiece(
        word_type_freqs=word_type_freqs,
        vocab_size=args.vocab_size,
        verbose=args.verbose_merges,
    )
    baseline_training_time = time.perf_counter() - start

    print("Final baseline vocabulary size:", len(baseline_vocabulary))
    print("Number of baseline merges:", len(baseline_merges))
    print("Baseline training time:", round(baseline_training_time, 4), "seconds")
    save_vocab(baseline_vocabulary, output_dir / "wordpiece_baseline_vocab.txt")

    # 2. Load held-out eval split and compare tokenization implementations.
    test_corpus = load_corpus(args.test_path) if Path(args.test_path).exists() else []
    valid_corpus = load_corpus(args.valid_path) if Path(args.valid_path).exists() else []
    eval_corpus = test_corpus or valid_corpus

    eval_word_tokens = collect_word_tokens(eval_corpus, args.max_eval_word_tokens)
    eval_word_types = sorted(set(eval_word_tokens))

    print("\n2) Baseline vs fast WordPiece tokenization")
    print("Evaluation word tokens used:", len(eval_word_tokens))
    print("Evaluation word types:", len(eval_word_types))

    timing, baseline_outputs, _, same_outputs = time_tokenization_comparison(
        eval_word_tokens, baseline_vocabulary
    )
    baseline_metrics = evaluate_tokenizer_outputs(baseline_outputs)

    print("Fast output same as baseline on the same eval word tokens:", same_outputs)
    print("Baseline average subword tokens / word token:", round(baseline_metrics["average_subword_tokens_per_word_token"], 4))
    print("Baseline word-token UNK rate:", round(baseline_metrics["word_token_unk_rate"], 6))
    print("Baseline tokenization time:", round(timing["baseline_tokenization_time_seconds"], 4), "seconds")
    print("Fast tokenization time:", round(timing["fast_tokenization_time_seconds"], 4), "seconds")
    print("Tokenization speedup:", timing["speedup"])

    # 3. Export evaluation-friendly word segmentations.
    print("\n3) Export surface segmentations for evaluation")
    export_surface_segmentations(
        eval_word_types,
        baseline_vocabulary,
        output_dir / "wordpiece_surface_segmentations_for_eval.json",
        output_dir / "wordpiece_surface_segmentations_for_eval.csv",
    )

    # 4. Optional experimental fast vocabulary-learning comparison.
    print("\n4) Experimental fast WordPiece vocabulary learning")
    if args.max_fast_training_word_types == 0:
        fast_training_results: dict[str, object] = {"skipped": True}
        print("Skipped experimental fast training comparison.")
    else:
        limited_word_type_freqs = limit_word_type_freqs(
            word_type_freqs, args.max_fast_training_word_types
        )
        print(
            "Word types used for BOTH baseline and fast training comparison:",
            len(limited_word_type_freqs),
        )
        fast_training_results = time_baseline_and_fast_training(
            word_type_freqs=limited_word_type_freqs,
            vocab_size=args.vocab_size,
        )
        for key, value in fast_training_results.items():
            print(key, ":", value)

    # 5. Save combined result file.
    results: dict[str, object] = {
        "settings": {
            "train_path": args.train_path,
            "valid_path": args.valid_path,
            "test_path": args.test_path,
            "vocab_size": args.vocab_size,
            "max_train_lines": args.max_train_lines,
            "max_eval_word_tokens": args.max_eval_word_tokens,
            "max_fast_training_word_types": args.max_fast_training_word_types,
        },
        "corpus_statistics": {
            "num_train_lines": len(train_corpus),
            "num_training_word_types": len(word_type_freqs),
            "num_eval_word_tokens": len(eval_word_tokens),
            "num_eval_word_types": len(eval_word_types),
        },
        "baseline_training": {
            "training_time_seconds": baseline_training_time,
            "vocabulary_size": len(baseline_vocabulary),
            "number_of_merges": len(baseline_merges),
        },
        "tokenization_comparison": {
            "same_outputs": same_outputs,
            **timing,
            **baseline_metrics,
        },
        "experimental_fast_training": fast_training_results,
    }

    save_json(results, output_dir / "wordpiece_experiment_results.json")
    print("\nSaved results to:", output_dir)


if __name__ == "__main__":
    main()
