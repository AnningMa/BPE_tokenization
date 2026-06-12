from collections import Counter
from collections.abc import Iterable
import re


def load_corpus(path: str) -> list[str]:
    
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pre_tokenize(text: str) -> list[str]:
    
    text = text.lower()
    word_tokens = re.findall(r"[a-z0-9']+|[^a-z0-9'\s]", text)
    return word_tokens


def get_word_type_frequencies(corpus: Iterable[str]) -> dict[str, int]:
    
    word_type_freqs = Counter()
    for line in corpus:
        word_type_freqs.update(pre_tokenize(line))
    return dict(word_type_freqs)