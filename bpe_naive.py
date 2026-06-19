import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from base import BaseTokenizer
from utils import pre_tokenize, get_word_type_frequencies

def _merge_split(split: list[str], a: str, b: str, merged: str) -> list[str]:
    """Apply a single (a, b) -> merged operation to one split list."""
    i = 0
    new_split: list[str] = []
    while i < len(split):
        if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
            new_split.append(merged)
            i += 2
        else:
            new_split.append(split[i])
            i += 1
    return new_split


def _merge_pair(a: str, b: str, splits: dict[str, list[str]]) -> None:  # O(W·L)
    """Apply a single (a, b) -> merged operation to all splits in place."""
    merged = a + b
    for word, split in splits.items():
        splits[word] = _merge_split(split, a, b, merged)


class NaiveBPE(BaseTokenizer):

    def __init__(self):
        self.vocab:       list[str] = []
        self.merges:      list[tuple[str, str, str]] = []
        self._word_freqs: dict[str, int] = {}
        self._splits:     dict[str, list[str]] = {}

    def _init_splits(self):
        self._splits = {w: list(w) for w in self._word_freqs} #O(W·L) where W is the number of unique words and L is the average word length

    def _compute_pair_freqs(self) :
        pair_freqs = defaultdict(int)
        for word, freq in self._word_freqs.items(): 
            split = self._splits[word]
            for i in range(len(split) - 1): #O(W·L) where W is the number of unique words and L is the average word length
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

        while len(self.vocab) < vocab_size:  #循环执行M次，每次扫描所有为一次 --> O(M·W·L) 
            pair_freqs = self._compute_pair_freqs()
            if not pair_freqs: 
                break

            best = max(pair_freqs, key=pair_freqs.get) #O(p) where p is the number of unique pairs
            a, b = best
            merged = a + b

            self.merges.append((a, b, merged))
            self.vocab.append(merged)

            _merge_pair(a, b, self._splits)

    def tokenize(self, text): #O(M·L·T） where M is the number of merges, L is the average word length, and T is the number of tokens in the input text. In practice, this can be quite slow for large vocabularies and long texts.
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

    def tokenize_longest(self, text: str):
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

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "word_freqs_train_top5000.json"
    print(f"Loading word frequencies from {data_path} ...")
    with open(data_path, encoding="utf-8") as f:
        word_freqs = json.load(f)
    print(f"  {len(word_freqs):,} unique words, {sum(word_freqs.values()):,} total tokens")

    VOCAB_SIZE = 5000
    print(f"\nTraining NaiveBPE (vocab_size={VOCAB_SIZE}) ...")
    t0 = time.time()
    bpe = NaiveBPE()
    bpe.train(vocab_size=VOCAB_SIZE, word_freqs=word_freqs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(bpe.merges)} merges learned")
    print(f"  Last 10 merges: {bpe.merges[-10:]}")

    print()
    test_words = ["running", "tokenization", "unhappy", "cats", "unknown", "preprocessing"]
    for word in test_words:
        print(f"  {word!r:18} -> {bpe.tokenize(word)}")
