import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import BaseTokenizer
from utils import pre_tokenize, get_word_type_frequencies
from utils.heap import LazyHeap


def _merge_split(split, a, b, merged):
    i = 0
    new_split = []
    while i < len(split):
        if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
            new_split.append(merged)
            i += 2
        else:
            new_split.append(split[i])
            i += 1
    return new_split


class FastBPE(BaseTokenizer):
    def __init__(self):
        self.vocab = []
        self.merges = {}
        self._word_freqs = {}
        self._splits = {}
        self._inv_index = {}
        self._heap = LazyHeap()
        self._tokenize_cache = {}

    def train(self, vocab_size, *, corpus=None, word_freqs=None):
        if word_freqs is None:
            if corpus is None:
                raise ValueError("Provide either `corpus` or `word_freqs`.")
            word_freqs = get_word_type_frequencies(corpus)

        self._word_freqs = word_freqs
        self._tokenize_cache.clear()
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

        self.warm_cache(1000)

    def tokenize(self, text):
        if not self.vocab or not self.merges:
            raise RuntimeError("Call train() before tokenize().")

        tokens = []
        for word in pre_tokenize(text):
            cached = self._tokenize_cache.get(word)
            if cached is not None:
                tokens.extend(cached)
            else:
                result = self._tokenize_word(word)
                self._tokenize_cache[word] = result
                tokens.extend(result)

        return tokens

    def tokenize_longest(self, text):
        if not self.vocab:
            raise RuntimeError("Call train() before tokenize_longest().")

        vocab = set(self.vocab)
        tokens = []

        for word in pre_tokenize(text):
            word_with_end = word + "_"
            i = 0
            while i < len(word_with_end):
                matched = None
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

        return tokens

    def _tokenize_word(self, word):
        vocab_set = set(self.vocab)
        split = [ch if ch in vocab_set else "<unk>" for ch in word] + ["_"]

        for (a, b), merged in self.merges.items():
            split = _merge_split(split, a, b, merged)

        return split

    def warm_cache(self, top_n=1000):
        if not self._word_freqs:
            return

        top_words = sorted(
            self._word_freqs.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]

        for word, _ in top_words:
            if word not in self._tokenize_cache:
                self._tokenize_cache[word] = self._tokenize_word(word)

    def _init_splits(self):
        self._splits = {w: list(w) + ["_"] for w in self._word_freqs}

    def _build_index(self):
        self._inv_index = {}
        pair_freqs = {}

        for word, freq in self._word_freqs.items():
            split = self._splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                self._inv_index.setdefault(pair, set()).add(word)

        self._heap = LazyHeap()
        for pair, freq in pair_freqs.items():
            self._heap.push(pair, freq)

    def _apply_merge(self, a, b, merged):
        affected = self._inv_index.pop((a, b), set())

        for word in affected:
            freq = self._word_freqs[word]
            old_split = self._splits[word]
            new_split = []
            delta = {}
            i = 0

            while i < len(old_split):
                if i < len(old_split) - 1 and old_split[i] == a and old_split[i + 1] == b:
                    if new_split:
                        left = new_split[-1]
                        delta[(left, a)] = delta.get((left, a), 0) - freq
                        delta[(left, merged)] = delta.get((left, merged), 0) + freq

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

            new_pair_set = {
                (new_split[j], new_split[j + 1])
                for j in range(len(new_split) - 1)
            }

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
