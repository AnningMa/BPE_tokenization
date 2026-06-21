import os
import re
import sys
from itertools import islice

# from bpe_naive import word_freqs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter

from datasets import load_dataset, load_from_disk

from utils import get_word_type_frequencies, pre_tokenize
from utils.heap import LazyHeap


class FastBPE:
    """
    BPE with Priority Queue + Inverted Index.

    Speedup over NaiveBPE:
      - Priority queue (LazyHeap): best pair in O(log P) instead of O(P).
      - Inverted index: each merge only visits words that contain the merged
        pair, instead of scanning the entire corpus.

    Per-merge cost: O(k * L * log P)
      k = number of distinct words containing the pair
      L = average length of those words
      P = number of distinct pairs currently in the heap
    vs. NaiveBPE: O(N) per merge (N = total corpus tokens)
    """

    def __init__(self):
        self.vocab: list[str] = []
        self.merges: dict[tuple[str, str], str] = {}

        self._word_freqs: dict[str, int] = {}
        self._splits: dict[str, list[str]] = {}
        self._inv_index: dict[tuple[str, str], set[str]] = {}
        self._heap: LazyHeap = LazyHeap()

    def train(
        self,
        vocab_size: int,
        *,
        corpus=None,
        word_freqs=None,
    ):
        if word_freqs is None:
            if corpus is None:
                raise ValueError("Provide either `corpus` or `word_freqs`.")
            word_freqs = get_word_type_frequencies(corpus)

        self._word_freqs = word_freqs
        alphabet = sorted({ch for word in word_freqs for ch in word})
        self.vocab = ["<unk>", "_"] + alphabet
        self.merges = {}

        if vocab_size <= len(self.vocab):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be greater than the initial "
                f"character vocabulary size ({len(self.vocab)})."
            )

        self._init_splits()
        self._build_index()

        while len(self.vocab) < vocab_size:
            result = self._heap.pop_best()
            if result is None:
                break
            best_pair, _ = result
            a, b = best_pair
            merged = a + b
            self.merges[best_pair] = merged
            self.vocab.append(merged)
            self._apply_merge(a, b, merged)

    def load_vocab(self, vo, me):
        with open(vo, "r") as f:
            for w in f:
                self.vocab.append(w.strip())
        with open(me, "r") as f:
            for line in f:
                pieces = line.split(" ")
                a = pieces[0].strip()
                b = pieces[1].strip()
                c = pieces[3].strip()
                self.merges[(a, b)] = c

    def tokenize(self, text: str) -> list[str]:
        if not self.vocab or not self.merges:
            raise RuntimeError("Call train() before tokenize().")

        vocab_set = set(self.vocab)
        splits: list[list[str]] = [
            [ch if ch in vocab_set else "<unk>" for ch in word] + ["_"]
            for word in pre_tokenize(text)
        ]

        for (a, b), merged in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                new_split: list[str] = []
                while i < len(split):
                    if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[idx] = new_split

        return [tok.replace("_", "") for split in splits for tok in split if tok != "_"]

    def tokenize_longest(self, text):

        if not self.vocab:
            raise RuntimeError("Call train() before tokenize_longest().")

        vocab = set(self.vocab)
        tokens: list[str] = []

        for word in pre_tokenize(text):
            word_with_end = word + "_"
            i = 0
            while i < len(word_with_end):
                matched: str | None = None
                for j in range(len(word_with_end), i, -1):
                    sub = word_with_end[i:j]
                    if sub in vocab:
                        matched = sub
                        break

                if matched is None:
                    tokens.append("<unk>")
                    i += 1
                else:
                    tokens.append(matched)
                    i += len(matched)

        tokens = [token.replace("_", "") for token in tokens if token != "_"]
        return tokens

    def _init_splits(self) -> None:
        self._splits = {w: list(w) + ["_"] for w in self._word_freqs}

    def _build_index(self) -> None:
        """One-time O(N) scan to build pair frequencies + inverted index + heap."""
        self._inv_index = {}
        pair_freqs: dict[tuple[str, str], int] = {}

        for word, freq in self._word_freqs.items():
            split = self._splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                self._inv_index.setdefault(pair, set()).add(word)

        self._heap = LazyHeap()
        for pair, freq in pair_freqs.items():
            self._heap.push(pair, freq)

    def _apply_merge(self, a: str, b: str, merged: str) -> None:

        affected = self._inv_index.pop((a, b), set())

        for word in affected:
            freq = self._word_freqs[word]
            old_split = self._splits[word]
            new_split: list[str] = []
            delta: dict[tuple[str, str], int] = {}
            i = 0

            while i < len(old_split):
                if (
                    i < len(old_split) - 1
                    and old_split[i] == a
                    and old_split[i + 1] == b
                ):
                    # --- left-context pair ---
                    if new_split:
                        left = new_split[-1]
                        delta[(left, a)] = delta.get((left, a), 0) - freq
                        delta[(left, merged)] = delta.get((left, merged), 0) + freq

                    # --- right-context pair ---
                    if i + 2 < len(old_split):
                        right = old_split[i + 2]
                        delta[(b, right)] = delta.get((b, right), 0) - freq
                        delta[(merged, right)] = delta.get((merged, right), 0) + freq

                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(old_split[i])
                    i += 1

            self._splits[word] = new_split

            # Pairs present in the new split (for inv_index consistency)
            new_pair_set = {
                (new_split[j], new_split[j + 1]) for j in range(len(new_split) - 1)
            }

            # Apply deltas: update heap and inv_index
            for pair, d in delta.items():
                if d == 0:
                    continue
                self._heap.update(pair, d)

                if pair in new_pair_set:
                    self._inv_index.setdefault(pair, set()).add(word)
                else:
                    bucket = self._inv_index.get(pair)
                    if bucket is not None:
                        bucket.discard(word)
                        if not bucket:
                            del self._inv_index[pair]


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

    # dataset = load_dataset(base_name, dataset_id, split=split)
    dataset = load_from_disk("../data/wikitext103")
    counter = Counter()

    for e in dataset:
        text = e["text"].strip().lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
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


_guten_cache = None


def guten_vocab(ds_id="manu/project_gutenberg", split="en"):
    global _guten_cache
    if _guten_cache is not None:
        return _guten_cache
    dataset = load_from_disk("../data/gutenberg_en")
    counter = Counter()
    subset_1k2 = [item["text"] for item in islice(dataset, 600)]

    for e in subset_1k2:
        text = e.strip().lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        counter.update(words)

    return counter


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    # data_path = Path(__file__).parent.parent / "data" / "word_freqs_train_top5000.json"
    # print(f"Loading word frequencies from {data_path} ...")
    # with open(data_path, encoding="utf-8") as f:
    #    word_freqs = json.load(f)

    word_freqs = make_vocab()  # guten_vocab()  # make_vocab()

    print(
        f"  {len(word_freqs):,} unique words, {sum(word_freqs.values()):,} total tokens"
    )

    VOCAB_SIZE = 5_000
    print(f"\nTraining FastBPE (vocab_size={VOCAB_SIZE}) ...")
    t0 = time.time()
    bpe = FastBPE()
    bpe.train(vocab_size=VOCAB_SIZE, word_freqs=word_freqs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(bpe.merges)} merges learned")
    print(f"  Last 10 merges: {list(bpe.merges.items())[-10:]}")

    print()
    test_words = [
        "running",
        "tokenization",
        "unhappy",
        "cats",
        "unknown",
        "preprocessing",
    ]
    for word in test_words:
        print(f"  {word!r:18} -> {bpe.tokenize(word)}")

    TRAIN_DATA = "wiki"

    with open(f"../data/bpe_{TRAIN_DATA}_v{VOCAB_SIZE}_vocab.txt", "w") as f:
        for w in bpe.vocab:
            f.write(w)
            f.write("\n")
    with open(f"../data/bpe_{TRAIN_DATA}_v{VOCAB_SIZE}_merges.txt", "w") as f:
        for (a, b), c in bpe.merges.items():
            f.write(f"{a} {b} -> {c}")
            f.write("\n")
