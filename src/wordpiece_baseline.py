from collections import Counter, defaultdict
import csv
import re
import time
from typing import Dict, List, Tuple, Set, Iterable, Optional, Any

# Phase 0: corpus loading and pre-tokenization

def load_corpus(path: str) -> List[str]:
    
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pre_tokenize(text: str) -> List[str]:
    
    text = text.lower()
    word_tokens = re.findall(r"[a-z]+", text)
    return word_tokens


def get_word_type_frequencies(corpus: Iterable[str]) -> Dict[str, int]:
    """
    Count word token occurrences and return frequencies for each word type.
    Returns:
        A dictionary mapping each word type to its frequency in the corpus.
        Example: {"happy": 3, "unhappy": 2}
    """
    word_type_freqs = Counter()
    for line in corpus:
        word_tokens = pre_tokenize(line)
        word_type_freqs.update(word_tokens)
    return dict(word_type_freqs)

# Phase 1: WordPiece vocabulary learning

def split_word_type_into_initial_subwords(word_type: str) -> List[str]:
    """
    Split a word type into initial character-level WordPiece subword types.
    Example:
        "word" -> ["w", "##o", "##r", "##d"]
    """
    if not word_type:
        return []
    return [word_type[0]] + ["##" + char for char in word_type[1:]]


def initialize_wordpiece_vocabulary(
    word_type_freqs: Dict[str, int],
    special_tokens: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Initialize the WordPiece vocabulary and the current split of each word type.
    Returns:
        vocabulary_list:
            Initial list of subword types, including special tokens and character-level WordPiece symbols.
        word_type_splits:
            Mapping from each word type to its current sequence of subword types.
    """
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    vocabulary_list = list(special_tokens)
    alphabet: Set[str] = set()
    word_type_splits: Dict[str, List[str]] = {}

    for word_type in word_type_freqs:
        split = split_word_type_into_initial_subwords(word_type)
        word_type_splits[word_type] = split
        alphabet.update(split)

    vocabulary_list.extend(sorted(alphabet))
    return vocabulary_list, word_type_splits


def compute_pair_scores(
    word_type_splits: Dict[str, List[str]],
    word_type_freqs: Dict[str, int],
) -> Dict[Tuple[str, str], float]:
    """
    Compute WordPiece scores for all adjacent subword pairs.
    score(pair) = pair_frequency / (frequency_of_first_subword * frequency_of_second_subword)
    This is the baseline implementation: after every merge, frequencies and scores are recomputed over all word types.
    This is expensive for large corpora.
    """
    subword_type_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)

    for word_type, split in word_type_splits.items():
        word_frequency = word_type_freqs[word_type]

        for subword_type in split:
            subword_type_freqs[subword_type] += word_frequency

        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += word_frequency

    scores: Dict[Tuple[str, str], float] = {}
    for pair, pair_frequency in pair_freqs.items():
        first_subword, second_subword = pair
        scores[pair] = pair_frequency / (
            subword_type_freqs[first_subword] * subword_type_freqs[second_subword]
        )

    return scores


def merge_pair_in_word_type_splits(
    first_subword: str,
    second_subword: str,
    word_type_splits: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], str]:
    """
    Merge a selected adjacent pair in all current word-type splits.
    Examples:
        "h" + "##a" -> "ha"
        "##p" + "##y" -> "##py"
    """
    if second_subword.startswith("##"):
        new_subword_type = first_subword + second_subword[2:]
    else:
        new_subword_type = first_subword + second_subword

    updated_word_type_splits: Dict[str, List[str]] = {}

    for word_type, split in word_type_splits.items():
        updated_split: List[str] = []
        i = 0

        while i < len(split):
            if (
                i < len(split) - 1
                and split[i] == first_subword
                and split[i + 1] == second_subword
            ):
                updated_split.append(new_subword_type)
                i += 2
            else:
                updated_split.append(split[i])
                i += 1

        updated_word_type_splits[word_type] = updated_split

    return updated_word_type_splits, new_subword_type


def train_wordpiece(
    word_type_freqs: Dict[str, int],
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[Set[str], Dict[str, List[str]], List[Tuple[str, str, str]]]:
    """
    Train a baseline WordPiece vocabulary. This function is the vocabulary learning phase, not the tokenization phase.
    Args:
        word_type_freqs: Frequencies of unique pre-tokenized word types in the training corpus.
        vocab_size: Target number of subword types in the final vocabulary.
        special_tokens: Optional list of special tokens.
        verbose: Whether to print merge information.
    Returns:
        vocabulary: Final set of subword types.
        word_type_splits: Final split of each training word type.
        merges: List of performed merges, useful for debugging and reporting.
    """
    vocabulary_list, word_type_splits = initialize_wordpiece_vocabulary(
        word_type_freqs, special_tokens
    )
    vocabulary: Set[str] = set(vocabulary_list)
    merges: List[Tuple[str, str, str]] = []

    while len(vocabulary) < vocab_size:
        scores = compute_pair_scores(word_type_splits, word_type_freqs)

        if not scores:
            break

        best_pair = max(scores, key=scores.get)
        first_subword, second_subword = best_pair

        word_type_splits, new_subword_type = merge_pair_in_word_type_splits(
            first_subword, second_subword, word_type_splits
        )

        if new_subword_type in vocabulary:
            break

        vocabulary.add(new_subword_type)
        merges.append((first_subword, second_subword, new_subword_type))

        if verbose:
            print(
                f"Merge {len(merges)}: "
                f"{first_subword} + {second_subword} -> {new_subword_type}"
            )

    return vocabulary, word_type_splits, merges


# Phase 2: WordPiece tokenization / inference

def encode_word_type(
    word_type: str,
    vocabulary: Set[str],
    unk_token: str = "[UNK]",
) -> List[str]:
    """
    Tokenize one pre-tokenized word using greedy longest-match-first.
    This is the WordPiece tokenization phase. It uses the final vocabulary learned in Phase 1 and resolves segmentation ambiguity by selecting the
    longest valid subword from left to right.
    If no valid segmentation is found, return [UNK] for the whole word.
    """
    word_type = word_type.lower()
    subword_tokens: List[str] = []
    start = 0

    while start < len(word_type):
        end = len(word_type)
        current_subword: Optional[str] = None

        while start < end:
            candidate = word_type[start:end]
            if start > 0:
                candidate = "##" + candidate

            if candidate in vocabulary:
                current_subword = candidate
                break

            end -= 1

        if current_subword is None:
            return [unk_token]

        subword_tokens.append(current_subword)
        start = end

    return subword_tokens


def tokenize_wordpiece(text: str, vocabulary: Set[str]) -> List[str]:
    """
    Tokenize a text into WordPiece subword tokens.
    Pre-tokenize the text into word tokens and tokenize each word token into WordPiece subword tokens.
    """
    output_subword_tokens: List[str] = []

    for word_token in pre_tokenize(text):
        output_subword_tokens.extend(encode_word_type(word_token, vocabulary))

    return output_subword_tokens


# Timing and simple tokenizer evaluation

def evaluate_wordpiece_on_word_types(
    test_word_types: List[str],
    vocabulary: Set[str],
) -> Dict[str, float]:
    """
    Evaluate tokenization on a list of unique word types.
    Metrics:
        tokenization_time_seconds: Total time for encoding the list.
        average_subword_tokens_per_word_type: Average number of subword tokens produced per word type.
        word_type_unk_rate: Proportion of word types encoded as [UNK].
        word_type_single_subword_rate: Proportion of word types kept as one subword token.
    """
    start_time = time.perf_counter()
    tokenized_word_types = [encode_word_type(word, vocabulary) for word in test_word_types]
    tokenization_time = time.perf_counter() - start_time

    total_word_types = len(test_word_types)
    total_subword_tokens = sum(len(tokens) for tokens in tokenized_word_types)
    unk_word_types = sum(1 for tokens in tokenized_word_types if tokens == ["[UNK]"])
    single_subword_word_types = sum(1 for tokens in tokenized_word_types if len(tokens) == 1)

    return {
        "tokenization_time_seconds": tokenization_time,
        "average_subword_tokens_per_word_type": (
            total_subword_tokens / total_word_types if total_word_types else 0.0
        ),
        "word_type_unk_rate": unk_word_types / total_word_types if total_word_types else 0.0,
        "word_type_single_subword_rate": (
            single_subword_word_types / total_word_types if total_word_types else 0.0
        ),
    }


def evaluate_wordpiece_on_text(
    texts: Iterable[str],
    vocabulary: Set[str],
) -> Dict[str, float]:
    """
    Evaluate tokenization on running text, using word tokens as occurrences.
    This complements type-level evaluation. A frequent word contributes more than a rare word because we count occurrences in the text.
    """
    word_tokens: List[str] = []
    for text in texts:
        word_tokens.extend(pre_tokenize(text))

    start_time = time.perf_counter()
    tokenized_word_tokens = [encode_word_type(word, vocabulary) for word in word_tokens]
    tokenization_time = time.perf_counter() - start_time

    total_word_tokens = len(word_tokens)
    total_subword_tokens = sum(len(tokens) for tokens in tokenized_word_tokens)
    unk_word_tokens = sum(1 for tokens in tokenized_word_tokens if tokens == ["[UNK]"])
    single_subword_word_tokens = sum(1 for tokens in tokenized_word_tokens if len(tokens) == 1)

    return {
        "tokenization_time_seconds": tokenization_time,
        "number_of_word_tokens": total_word_tokens,
        "number_of_subword_tokens": total_subword_tokens,
        "average_subword_tokens_per_word_token": (
            total_subword_tokens / total_word_tokens if total_word_tokens else 0.0
        ),
        "word_token_unk_rate": unk_word_tokens / total_word_tokens if total_word_tokens else 0.0,
        "word_token_single_subword_rate": (
            single_subword_word_tokens / total_word_tokens if total_word_tokens else 0.0
        ),
    }


def time_training(
    word_type_freqs: Dict[str, int],
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
) -> Tuple[Set[str], Dict[str, List[str]], List[Tuple[str, str, str]], float]:
    """
    Train WordPiece and return the training time.
    Useful for comparing baseline and fast vocabulary-learning implementations.
    """
    start_time = time.perf_counter()
    vocabulary, word_type_splits, merges = train_wordpiece(
        word_type_freqs, vocab_size, special_tokens=special_tokens
    )
    training_time = time.perf_counter() - start_time
    return vocabulary, word_type_splits, merges, training_time


# Saving outputs

def save_vocab(vocabulary: Set[str], path: str) -> None:
    
    with open(path, "w", encoding="utf-8") as f:
        for subword_type in sorted(vocabulary):
            f.write(subword_type + "\n")


def save_tokenization_examples(
    test_word_types: List[str],
    vocabulary: Set[str],
    path: str,
) -> None:
   
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word_type", "wordpiece_subword_tokens"])
        for word_type in test_word_types:
            subword_tokens = encode_word_type(word_type, vocabulary)
            writer.writerow([word_type, " ".join(subword_tokens)])


# Ambiguity demonstration for the report

def get_all_possible_segmentations(
    word_type: str,
    vocabulary: Set[str],
    unk_token: str = "[UNK]",
) -> List[List[str]]:
    """
    Enumerate all possible segmentations of one word type under a vocabulary.
    This function is not used by the tokenizer. It is useful for showing why an ambiguity-resolution strategy is needed. WordPiece resolves this ambiguity
    with greedy longest-match-first tokenization.
    """
    word_type = word_type.lower()
    results: List[List[str]] = []

    def backtrack(start: int, current: List[str]) -> None:
        if start == len(word_type):
            results.append(current.copy())
            return

        for end in range(start + 1, len(word_type) + 1):
            candidate = word_type[start:end]
            if start > 0:
                candidate = "##" + candidate
            if candidate in vocabulary:
                current.append(candidate)
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    return results if results else [[unk_token]]


def ambiguity_demo() -> Dict[str, Any]:
    """
    Return a small example showing segmentation ambiguity and its resolution.
    """
    demo_vocabulary = {
        "un", "##believable", "##believ", "##able", "unbelievable"
    }
    word_type = "unbelievable"
    return {
        "word_type": word_type,
        "all_possible_segmentations": get_all_possible_segmentations(
            word_type, demo_vocabulary
        ),
        "longest_match_first_segmentation": encode_word_type(
            word_type, demo_vocabulary
        ),
    }
