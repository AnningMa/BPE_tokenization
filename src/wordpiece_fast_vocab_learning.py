"""
Experimental faster WordPiece vocabulary learning using local updates.

This module optimizes the vocabulary-learning phase, not the tokenization phase.
It is intended as an experimental alternative to the baseline training loop, which recomputes all subword frequencies, pair frequencies, and pair scores
over all word types after every merge.

Main idea:
- Maintain current splits for each word type.
- Maintain global subword frequencies and adjacent-pair frequencies.
- Maintain an index from each adjacent pair to the word types currently containing it.
- Use a priority queue for WordPiece pair scores.
- After each merge, update only word types that contain the selected pair.

WordPiece scores depend on both pair frequencies and subword frequencies. A merge changes the frequencies of several subword types, so scores for pairs not directly
merged may also become stale. To keep the implementation correct, this version uses lazy heap validation: when a pair is popped from the heap, its score is
recomputed from the current statistics before it is accepted.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from utils import pre_tokenize, get_word_type_frequencies


Pair = tuple[str, str]
Merge = tuple[str, str, str]

SPECIAL_TOKENS = ["[UNK]"]
CONTINUATION_PREFIX = "##"


def split_word_type_into_initial_subwords(word_type: str) -> list[str]:
    
    if not word_type:
        return []
    return [word_type[0]] + [CONTINUATION_PREFIX + char for char in word_type[1:]]


def merge_subword_types(first_subword: str, second_subword: str) -> str:
    """Create the new WordPiece subword type obtained by merging two adjacent types."""
    if second_subword.startswith(CONTINUATION_PREFIX):
        return first_subword + second_subword[len(CONTINUATION_PREFIX):]
    return first_subword + second_subword


def iter_adjacent_pairs(split: Sequence[str]) -> Iterable[Pair]:
    """Yield adjacent subword pairs from a word-type split."""
    for i in range(len(split) - 1):
        yield split[i], split[i + 1]


def pair_score(
    pair: Pair,
    pair_freqs: dict[Pair, int],
    subword_type_freqs: dict[str, int],
) -> float:
    """Compute the current WordPiece score for one adjacent pair."""
    pair_frequency = pair_freqs.get(pair, 0)
    if pair_frequency <= 0:
        return 0.0

    first_subword, second_subword = pair
    first_freq = subword_type_freqs.get(first_subword, 0)
    second_freq = subword_type_freqs.get(second_subword, 0)
    if first_freq <= 0 or second_freq <= 0:
        return 0.0

    return pair_frequency / (first_freq * second_freq)


def initialize_training_state(
    word_type_freqs: dict[str, int],
    special_tokens: list[str] | None = None,
) -> tuple[
    set[str],
    dict[str, list[str]],
    defaultdict[str, int],
    defaultdict[Pair, int],
    defaultdict[Pair, set[str]],
    list[tuple[float, Pair]],
]:
    """
    Initialize vocabulary, word-type splits, frequency tables, indexes, and heap.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    vocabulary: set[str] = set(special_tokens)
    word_type_splits: dict[str, list[str]] = {}
    subword_type_freqs: defaultdict[str, int] = defaultdict(int)
    pair_freqs: defaultdict[Pair, int] = defaultdict(int)
    pair_to_word_types: defaultdict[Pair, set[str]] = defaultdict(set)

    for word_type, word_frequency in word_type_freqs.items():
        split = split_word_type_into_initial_subwords(word_type)
        word_type_splits[word_type] = split
        vocabulary.update(split)

        for subword_type in split:
            subword_type_freqs[subword_type] += word_frequency

        for pair in iter_adjacent_pairs(split):
            pair_freqs[pair] += word_frequency
            pair_to_word_types[pair].add(word_type)

    heap: list[tuple[float, Pair]] = []
    for pair in pair_freqs:
        score = pair_score(pair, pair_freqs, subword_type_freqs)
        if score > 0:
            heapq.heappush(heap, (-score, pair))

    return (
        vocabulary,
        word_type_splits,
        subword_type_freqs,
        pair_freqs,
        pair_to_word_types,
        heap,
    )


def subtract_word_type_statistics(
    word_type: str,
    split: Sequence[str],
    word_frequency: int,
    subword_type_freqs: defaultdict[str, int],
    pair_freqs: defaultdict[Pair, int],
    pair_to_word_types: defaultdict[Pair, set[str]],
) -> set[Pair]:
    """Remove one word type's old split from the global statistics."""
    changed_pairs: set[Pair] = set(iter_adjacent_pairs(split))

    for subword_type in split:
        subword_type_freqs[subword_type] -= word_frequency
        if subword_type_freqs[subword_type] <= 0:
            del subword_type_freqs[subword_type]

    for pair in changed_pairs:
        pair_freqs[pair] -= word_frequency
        if pair_freqs[pair] <= 0:
            del pair_freqs[pair]
        pair_to_word_types[pair].discard(word_type)
        if not pair_to_word_types[pair]:
            del pair_to_word_types[pair]

    return changed_pairs


def add_word_type_statistics(
    word_type: str,
    split: Sequence[str],
    word_frequency: int,
    subword_type_freqs: defaultdict[str, int],
    pair_freqs: defaultdict[Pair, int],
    pair_to_word_types: defaultdict[Pair, set[str]],
) -> set[Pair]:
    """Add one word type's new split to the global statistics."""
    changed_pairs: set[Pair] = set(iter_adjacent_pairs(split))

    for subword_type in split:
        subword_type_freqs[subword_type] += word_frequency

    for pair in changed_pairs:
        pair_freqs[pair] += word_frequency
        pair_to_word_types[pair].add(word_type)

    return changed_pairs


def merge_pair_in_split(split: Sequence[str], pair_to_merge: Pair, new_subword_type: str) -> List[str]:
    """Merge all non-overlapping occurrences of a pair inside one word-type split."""
    first_subword, second_subword = pair_to_merge
    updated_split: list[str] = []
    i = 0

    while i < len(split):
        if i < len(split) - 1 and split[i] == first_subword and split[i + 1] == second_subword:
            updated_split.append(new_subword_type)
            i += 2
        else:
            updated_split.append(split[i])
            i += 1

    return updated_split


def push_current_score(
    pair: Pair,
    heap: list[tuple[float, Pair]],
    pair_freqs: dict[Pair, int],
    subword_type_freqs: dict[str, int],
) -> None:
    """Push a pair's current score to the heap if the pair is still active."""
    score = pair_score(pair, pair_freqs, subword_type_freqs)
    if score > 0:
        heapq.heappush(heap, (-score, pair))


def pop_best_current_pair(
    heap: list[tuple[float, Pair]],
    pair_freqs: dict[Pair, int],
    subword_type_freqs: dict[str, int],
    tolerance: float = 1e-15,
) -> Pair | None:
    """
    Pop the highest-scoring currently valid pair using lazy validation.

    Heap entries can become stale after local updates. When we pop an entry, we
    recompute its current score. If the stored score is stale, we push the
    current score and continue. A pair is accepted only when the heap entry still
    matches the current statistics.
    """
    while heap:
        stored_negative_score, pair = heapq.heappop(heap)
        stored_score = -stored_negative_score
        current_score = pair_score(pair, pair_freqs, subword_type_freqs)

        if current_score <= 0:
            continue

        if abs(stored_score - current_score) <= tolerance:
            return pair

        heapq.heappush(heap, (-current_score, pair))

    return None


def train_wordpiece_fast_local_updates(
    word_type_freqs: dict[str, int],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
) -> tuple[set[str], dict[str, list[str]], list[Merge]]:
    """
    Train WordPiece vocabulary with experimental local updates.

    This aims to reduce repeated full-corpus recomputation. It should produce the
    same kind of vocabulary as baseline WordPiece training, but the exact merge
    order can differ if there are score ties, because heaps do not guarantee the
    same tie-breaking as a dictionary max scan.
    """
    (
        vocabulary,
        word_type_splits,
        subword_type_freqs,
        pair_freqs,
        pair_to_word_types,
        heap,
    ) = initialize_training_state(word_type_freqs, special_tokens)

    merges: list[Merge] = []

    while len(vocabulary) < vocab_size:
        best_pair = pop_best_current_pair(heap, pair_freqs, subword_type_freqs)
        if best_pair is None:
            break

        first_subword, second_subword = best_pair
        new_subword_type = merge_subword_types(first_subword, second_subword)

        if new_subword_type in vocabulary:
            # This can happen in unusual cases. Skip this merge to avoid a loop.
            pair_freqs.pop(best_pair, None)
            pair_to_word_types.pop(best_pair, None)
            continue

        affected_word_types = list(pair_to_word_types.get(best_pair, set()))
        if not affected_word_types:
            continue

        changed_pairs: set[Pair] = set()

        # Update only word types that currently contain the selected pair.
        for word_type in affected_word_types:
            old_split = word_type_splits[word_type]
            word_frequency = word_type_freqs[word_type]

            changed_pairs.update(
                subtract_word_type_statistics(
                    word_type,
                    old_split,
                    word_frequency,
                    subword_type_freqs,
                    pair_freqs,
                    pair_to_word_types,
                )
            )

            new_split = merge_pair_in_split(old_split, best_pair, new_subword_type)
            word_type_splits[word_type] = new_split

            changed_pairs.update(
                add_word_type_statistics(
                    word_type,
                    new_split,
                    word_frequency,
                    subword_type_freqs,
                    pair_freqs,
                    pair_to_word_types,
                )
            )

        vocabulary.add(new_subword_type)
        merges.append((first_subword, second_subword, new_subword_type))

        # Scores for changed pairs must be pushed. Scores for other pairs can also
        # be affected if they contain subwords whose frequencies changed. The lazy
        # validation in pop_best_current_pair handles stale heap entries, and the
        # directly changed pairs are the most important ones to refresh promptly.
        for pair in changed_pairs:
            push_current_score(pair, heap, pair_freqs, subword_type_freqs)

        # Also refresh currently active pairs that contain the newly created subword.
        # This is small compared with scanning all word types and helps the heap
        # find newly possible high-scoring pairs sooner.
        for pair in list(pair_freqs.keys()):
            if new_subword_type in pair:
                push_current_score(pair, heap, pair_freqs, subword_type_freqs)

        if verbose:
            print(
                f"Merge {len(merges)}: {first_subword} + {second_subword} -> "
                f"{new_subword_type} | affected_word_types={len(affected_word_types)}"
            )

    return vocabulary, word_type_splits, merges


def time_fast_training(
    word_type_freqs: dict[str, int],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[set[str], dict[str, list[str]], list[Merge], float]:
    """Train with local updates and return the elapsed time."""
    start_time = time.perf_counter()
    vocabulary, word_type_splits, merges = train_wordpiece_fast_local_updates(
        word_type_freqs, vocab_size, special_tokens=special_tokens
    )
    elapsed = time.perf_counter() - start_time
    return vocabulary, word_type_splits, merges, elapsed


# Optional baseline functions for quick local comparison.

def compute_pair_scores_baseline(
    word_type_splits: dict[str, list[str]],
    word_type_freqs: dict[str, int],
) -> dict[Pair, float]:
    """Baseline full recomputation of WordPiece pair scores."""
    subword_type_freqs: defaultdict[str, int] = defaultdict(int)
    pair_freqs: defaultdict[Pair, int] = defaultdict(int)

    for word_type, split in word_type_splits.items():
        word_frequency = word_type_freqs[word_type]
        for subword_type in split:
            subword_type_freqs[subword_type] += word_frequency
        for pair in iter_adjacent_pairs(split):
            pair_freqs[pair] += word_frequency

    return {
        pair: freq / (subword_type_freqs[pair[0]] * subword_type_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }


def merge_pair_in_all_word_type_splits(
    pair_to_merge: Pair,
    word_type_splits: dict[str, list[str]],
) -> tuple[dict[str, list[str]], str]:
    """Baseline merge over all word types."""
    new_subword_type = merge_subword_types(*pair_to_merge)
    return {
        word_type: merge_pair_in_split(split, pair_to_merge, new_subword_type)
        for word_type, split in word_type_splits.items()
    }, new_subword_type


def train_wordpiece_baseline_for_comparison(
    word_type_freqs: dict[str, int],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[set[str], dict[str, list[str]], list[Merge]]:
    """Simple baseline WordPiece training for timing comparison."""
    vocabulary, word_type_splits, *_ = initialize_training_state(word_type_freqs, special_tokens)
    merges: list[Merge] = []

    while len(vocabulary) < vocab_size:
        scores = compute_pair_scores_baseline(word_type_splits, word_type_freqs)
        if not scores:
            break
        best_pair = max(scores, key=scores.get)
        word_type_splits, new_subword_type = merge_pair_in_all_word_type_splits(
            best_pair, word_type_splits
        )
        if new_subword_type in vocabulary:
            break
        vocabulary.add(new_subword_type)
        merges.append((best_pair[0], best_pair[1], new_subword_type))

    return vocabulary, word_type_splits, merges


def time_baseline_and_fast_training(
    word_type_freqs: dict[str, int],
    vocab_size: int,
) -> dict[str, object]:
    """Compare baseline full-recomputation training and local-update training."""
    start = time.perf_counter()
    baseline_vocab, _, baseline_merges = train_wordpiece_baseline_for_comparison(
        word_type_freqs, vocab_size
    )
    baseline_time = time.perf_counter() - start

    start = time.perf_counter()
    fast_vocab, _, fast_merges = train_wordpiece_fast_local_updates(
        word_type_freqs, vocab_size
    )
    fast_time = time.perf_counter() - start

    return {
        "baseline_training_time_seconds": baseline_time,
        "fast_training_time_seconds": fast_time,
        "speedup": baseline_time / fast_time if fast_time > 0 else None,
        "baseline_vocab_size": len(baseline_vocab),
        "fast_vocab_size": len(fast_vocab),
        "baseline_num_merges": len(baseline_merges),
        "fast_num_merges": len(fast_merges),
        "same_merge_sequence": baseline_merges == fast_merges,
        "same_vocabulary": baseline_vocab == fast_vocab,
    }
