import re
from collections import defaultdict
from typing import Optional
from datasets import load_dataset


def pre_tokenize(text: str) -> list[str]:
    tokens = []
    for word in text.split():
        parts = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]", word)
        tokens.extend(parts)
    return tokens


def load_corpus(
    split: str = "train",
    max_sentences: Optional[int] = None,
    min_length: int = 10,
) -> list[str]:
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)

    sentences: list[str] = []
    for row in dataset:
        line: str = row["text"].strip()
        if not line or line.startswith(" = "):
            continue
        if len(line) < min_length:
            continue
        sentences.append(line)
        if max_sentences is not None and len(sentences) >= max_sentences:
            break
    return sentences


def get_word_type_frequencies(corpus: list[str]) -> dict[str, int]:
    word_freqs: dict[str, int] = defaultdict(int)
    for text in corpus:
        for word in pre_tokenize(text):
            word_freqs[word] += 1
    return dict(word_freqs)
