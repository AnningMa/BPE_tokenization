import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from base import BaseTokenizer
from utils import pre_tokenize, get_word_type_frequencies


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

def _merge_pair(a, b, splits):
    merged = a + b
    for word, split in splits.items():
        splits[word] = _merge_split(split, a, b, merged)


class NaiveBPE(BaseTokenizer):
    def __init__(self):
        self.vocab = []
        self.merges = []
        self._word_freqs = {}
        self._splits = {}

    def _init_splits(self):
        self._splits = {w: list(w) for w in self._word_freqs}

    def _compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self._word_freqs.items():
            split = self._splits[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i + 1])] += freq
        return pair_freqs

    def train(self, vocab_size, *, corpus=None, word_freqs=None):
        if word_freqs is None:
            if corpus is None:
                raise ValueError("Provide either `corpus` or `word_freqs`.")
            word_freqs = get_word_type_frequencies(corpus)

        self._word_freqs = word_freqs
        alphabet = sorted({ch for word in word_freqs for ch in word})
        self.vocab = ["<unk>"] + alphabet
        self.merges = []

        if vocab_size <= len(self.vocab):
            raise ValueError("vocab_size too small.")

        self._init_splits()

        while len(self.vocab) < vocab_size:
            pair_freqs = self._compute_pair_freqs()
            if not pair_freqs:
                break

            best = max(pair_freqs, key=pair_freqs.get)
            a, b = best
            merged = a + b

            self.merges.append((a, b, merged))
            self.vocab.append(merged)

            _merge_pair(a, b, self._splits)

    def tokenize(self, text):
        if not self.vocab or not self.merges:
            raise RuntimeError("Call train() first.")

        vocab_set = set(self.vocab)

        splits = [
            [ch if ch in vocab_set else "<unk>" for ch in word]
            for word in pre_tokenize(text)
        ]

        for a, b, merged in self.merges:
            splits = [_merge_split(split, a, b, merged) for split in splits]

        return [tok for split in splits for tok in split]

    def tokenize_longest(self, text):
        if not self.vocab:
            raise RuntimeError("Call train() first.")

        vocab = set(self.vocab)
        tokens = []

        for word in pre_tokenize(text):
            i = 0
            while i < len(word):
                matched = None
                for j in range(len(word), i, -1):
                    sub = word[i:j]
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
