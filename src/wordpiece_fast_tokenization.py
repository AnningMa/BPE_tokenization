"""
Fast WordPiece tokenization using trie-based longest-match-first.

The optimization here targets the WordPiece tokenization phase. The baseline
longest-match-first implementation repeatedly creates candidate substrings and
checks whether they are in the vocabulary. The fast implementation stores the
vocabulary in tries and finds the longest valid match by traversing characters.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


UNK_TOKEN = "[UNK]"
CONTINUATION_PREFIX = "##"


def pre_tokenize(text: str) -> List[str]:
    
    return re.findall(r"[a-z]+", text.lower())


@dataclass
class TrieNode:
    """A node in a trie storing WordPiece subword types."""

    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    subword_type: Optional[str] = None


class WordPieceTrieTokenizer:
    """
    Fast WordPiece tokenizer using two tries.
    We use two tries because WordPiece distinguishes between subwords that can begin a word and continuation subwords:
    - start_trie stores vocabulary items without the ## prefix.
    - continuation_trie stores vocabulary items with the ## prefix, but indexes them by their surface form without ##.
    This implements the same ambiguity-resolution strategy as the baseline
    WordPiece tokenizer: greedy longest-match-first from left to right.
    """

    def __init__(
        self,
        vocabulary: Set[str],
        unk_token: str = UNK_TOKEN,
        continuation_prefix: str = CONTINUATION_PREFIX,
    ) -> None:
        self.vocabulary = set(vocabulary)
        self.unk_token = unk_token
        self.continuation_prefix = continuation_prefix
        self.start_trie = TrieNode()
        self.continuation_trie = TrieNode()
        self._build_tries()

    @staticmethod
    def _insert_subword_type(root: TrieNode, surface: str, subword_type: str) -> None:
        """
        Insert one subword type into a trie.
        surface is the string used for character traversal, without ##.
        subword_type is the original vocabulary item, with ## if applicable.
        """
        node = root
        for char in surface:
            node = node.children.setdefault(char, TrieNode())
        node.subword_type = subword_type

    def _build_tries(self) -> None:
        """Build start and continuation tries from the final vocabulary."""
        for subword_type in self.vocabulary:
            # Special tokens such as [PAD], [UNK], [CLS] are not ordinary
            # character-level matches inside words.
            if subword_type.startswith("[") and subword_type.endswith("]"):
                continue

            if subword_type.startswith(self.continuation_prefix):
                surface = subword_type[len(self.continuation_prefix) :]
                if surface:
                    self._insert_subword_type(
                        self.continuation_trie, surface, subword_type
                    )
            else:
                self._insert_subword_type(self.start_trie, subword_type, subword_type)

    @staticmethod
    def _longest_match_from_trie(
        word_type: str,
        start_position: int,
        trie_root: TrieNode,
    ) -> Tuple[Optional[str], int]:
        """
        Return the longest subword type matching word_type[start_position:].
        If no vocabulary item matches from this position, returns (None, start_position).
        """
        node = trie_root
        best_subword_type: Optional[str] = None
        best_end_position = start_position
        position = start_position

        while position < len(word_type) and word_type[position] in node.children:
            node = node.children[word_type[position]]
            position += 1
            if node.subword_type is not None:
                best_subword_type = node.subword_type
                best_end_position = position

        return best_subword_type, best_end_position

    def encode_word_type(self, word_type: str) -> List[str]:
        """
        Segment one word type into WordPiece subword tokens.
        This is the fast version of greedy longest-match-first tokenization. It resolves segmentation ambiguity by selecting the longest valid subword
        type at each position.
        """
        word_type = word_type.lower()
        subword_tokens: List[str] = []
        start_position = 0

        while start_position < len(word_type):
            trie = self.start_trie if start_position == 0 else self.continuation_trie
            subword_type, end_position = self._longest_match_from_trie(
                word_type, start_position, trie
            )

            if subword_type is None:
                return [self.unk_token]

            subword_tokens.append(subword_type)
            start_position = end_position

        return subword_tokens

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize running text into WordPiece subword tokens.
        The text is first pre-tokenized into word tokens. Each word token is then segmented into subword tokens using the trie-based tokenizer.
        """
        output_subword_tokens: List[str] = []
        for word_token in pre_tokenize(text):
            output_subword_tokens.extend(self.encode_word_type(word_token))
        return output_subword_tokens


def encode_word_type_baseline(
    word_type: str,
    vocabulary: Set[str],
    unk_token: str = UNK_TOKEN,
    continuation_prefix: str = CONTINUATION_PREFIX,
) -> List[str]:
    """
    Baseline WordPiece longest-match-first tokenization.
    This version repeatedly creates candidate substrings from longest to shortest and checks whether they exist in the vocabulary. It is simple and useful for
    correctness checks, but less efficient than trie traversal.
    """
    word_type = word_type.lower()
    subword_tokens: List[str] = []
    start_position = 0

    while start_position < len(word_type):
        matched_subword: Optional[str] = None
        matched_end_position = start_position

        for end_position in range(len(word_type), start_position, -1):
            surface = word_type[start_position:end_position]
            candidate = (
                surface
                if start_position == 0
                else continuation_prefix + surface
            )
            if candidate in vocabulary:
                matched_subword = candidate
                matched_end_position = end_position
                break

        if matched_subword is None:
            return [unk_token]

        subword_tokens.append(matched_subword)
        start_position = matched_end_position

    return subword_tokens


def compare_baseline_and_fast_outputs(
    word_types: Sequence[str],
    vocabulary: Set[str],
) -> bool:
    """
    Check whether baseline and fast tokenization return identical outputs.
    The fast version should not change the tokenization result. It should only change the implementation speed.
    """
    fast_tokenizer = WordPieceTrieTokenizer(vocabulary)

    for word_type in word_types:
        baseline = encode_word_type_baseline(word_type, vocabulary)
        fast = fast_tokenizer.encode_word_type(word_type)
        if baseline != fast:
            print("Mismatch found.")
            print("word_type:", word_type)
            print("baseline:", baseline)
            print("fast:", fast)
            return False

    return True


def time_baseline_and_fast_tokenization(
    word_tokens: Sequence[str],
    vocabulary: Set[str],
) -> Dict[str, object]:
    """
    Measure tokenization time on the same sequence of word tokens.
    This compares only the tokenization phase. It does not include vocabulary learning time and does not include trie construction time in the fast timing,
    because the trie is built once from the final vocabulary and then reused.
    """
    fast_tokenizer = WordPieceTrieTokenizer(vocabulary)

    start_time = time.perf_counter()
    baseline_outputs = [
        encode_word_type_baseline(word_token, vocabulary) for word_token in word_tokens
    ]
    baseline_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    fast_outputs = [fast_tokenizer.encode_word_type(word_token) for word_token in word_tokens]
    fast_time = time.perf_counter() - start_time

    same_outputs = baseline_outputs == fast_outputs
    speedup = baseline_time / fast_time if fast_time > 0 else None

    return {
        "num_word_tokens": len(word_tokens),
        "baseline_tokenization_time_seconds": baseline_time,
        "fast_tokenization_time_seconds": fast_time,
        "speedup": speedup,
        "same_outputs": same_outputs,
    }


def evaluate_fast_tokenizer_on_word_tokens(
    word_tokens: Sequence[str],
    vocabulary: Set[str],
) -> Dict[str, float]:
    """Compute simple tokenization metrics on word-token occurrences."""
    fast_tokenizer = WordPieceTrieTokenizer(vocabulary)
    outputs = [fast_tokenizer.encode_word_type(word_token) for word_token in word_tokens]

    num_word_tokens = len(outputs)
    if num_word_tokens == 0:
        return {
            "average_subword_tokens_per_word_token": 0.0,
            "word_token_unk_rate": 0.0,
            "word_token_single_subword_rate": 0.0,
        }

    total_subword_tokens = sum(len(subword_tokens) for subword_tokens in outputs)
    unk_word_tokens = sum(1 for subword_tokens in outputs if subword_tokens == [UNK_TOKEN])
    single_subword_word_tokens = sum(
        1 for subword_tokens in outputs if len(subword_tokens) == 1 and subword_tokens != [UNK_TOKEN]
    )

    return {
        "average_subword_tokens_per_word_token": total_subword_tokens / num_word_tokens,
        "word_token_unk_rate": unk_word_tokens / num_word_tokens,
        "word_token_single_subword_rate": single_subword_word_tokens / num_word_tokens,
    }
