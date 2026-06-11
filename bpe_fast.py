import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import BaseTokenizer
from utils import pre_tokenize, get_word_freqs as get_word_type_frequencies
from utils.heap import LazyHeap


class FastBPE(BaseTokenizer):
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

    def __init__(self) -> None:
        self.vocab:  list[str] = []
        self.merges: dict[tuple[str, str], str] = {}

        self._word_freqs: dict[str, int] = {}
        self._splits:     dict[str, list[str]] = {}
        self._inv_index:  dict[tuple[str, str], set[str]] = {}
        self._heap:       LazyHeap = LazyHeap()

    def train(
        self,
        vocab_size: int,
        *,
        corpus: list[str] | None = None,
        word_freqs: dict[str, int] | None = None,
    ) -> None:
        if word_freqs is None:
            if corpus is None:
                raise ValueError("Provide either `corpus` or `word_freqs`.")
            word_freqs = get_word_type_frequencies(corpus)

        self._word_freqs = word_freqs
        alphabet = sorted({ch for word in word_freqs for ch in word})
        self.vocab  = ["<unk>"] + alphabet
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

    def tokenize(self, text: str) -> list[str]:
        if not self.vocab or not self.merges:
            raise RuntimeError("Call train() before tokenize().")

        vocab_set = set(self.vocab)
        splits: list[list[str]] = [
            [ch if ch in vocab_set else "<unk>" for ch in word]
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

        return [tok for split in splits for tok in split]

  
    def _init_splits(self) -> None:
        self._splits = {w: list(w) for w in self._word_freqs}

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
            freq      = self._word_freqs[word]
            old_split = self._splits[word]
            new_split: list[str] = []
            delta:     dict[tuple[str, str], int] = {}
            i = 0

            while i < len(old_split):
                if i < len(old_split) - 1 and old_split[i] == a and old_split[i + 1] == b:
                    # --- left-context pair ---
                    if new_split:
                        left = new_split[-1]
                        delta[(left, a)]      = delta.get((left, a), 0)      - freq
                        delta[(left, merged)] = delta.get((left, merged), 0) + freq

                    # --- right-context pair ---
                    if i + 2 < len(old_split):
                        right = old_split[i + 2]
                        delta[(b, right)]      = delta.get((b, right), 0)      - freq
                        delta[(merged, right)] = delta.get((merged, right), 0) + freq

                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(old_split[i])
                    i += 1

            self._splits[word] = new_split

            # Pairs present in the new split (for inv_index consistency)
            new_pair_set = {
                (new_split[j], new_split[j + 1])
                for j in range(len(new_split) - 1)
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


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "word_freqs_train_top5000.json"
    print(f"Loading word frequencies from {data_path} ...")
    with open(data_path, encoding="utf-8") as f:
        word_freqs = json.load(f)
    print(f"  {len(word_freqs):,} unique words, {sum(word_freqs.values()):,} total tokens")

    VOCAB_SIZE = 1000
    print(f"\nTraining FastBPE (vocab_size={VOCAB_SIZE}) ...")
    t0 = time.time()
    bpe = FastBPE()
    bpe.train(vocab_size=VOCAB_SIZE, word_freqs=word_freqs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(bpe.merges)} merges learned")
    print(f"  Last 10 merges: {list(bpe.merges.items())[-10:]}")

    print()
    test_words = ["running", "tokenization", "unhappy", "cats", "unknown", "preprocessing"]
    for word in test_words:
        print(f"  {word!r:18} -> {bpe.tokenize(word)}")
