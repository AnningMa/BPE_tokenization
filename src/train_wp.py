import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from wordpiece_baseline import encode_word_type, save_vocab, train_wordpiece
from wordpiece_fast_vocab_learning import time_fast_training

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


SAVE_VOCAB_PATH = Path(__file__).parent.parent / "data"
vocab = make_vocab()
wp_voc, _, _, _ = time_fast_training(vocab, vocab_size=20_000, min_pair_freq=500)
save_vocab(wp_voc, str(SAVE_VOCAB_PATH / "wiki_v20000_m500_vocab.txt"))
