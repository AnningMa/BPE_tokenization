### install from requirements.txt

`pip install -r requirements.txt`

### load data

```
python prepare_data.py
```
This script downloads:
data/wikitext-103/
  wiki.train.tokens
  wiki.valid.tokens
  wiki.test.tokens
data/gutenberg/
  gutenberg_600_train.txt
  gutenberg_1200_train.txt
  (The Gutenberg test set is already included in this repository: data/guten_test_chunk.txt)
  
### normalization
Before training and evaluation, the text is lowercased and pre-tokenized into word-level units.
We use r"[a-z]+(?:'[a-z]+)?" which keeps alphabetic English words and simple apostrophe forms such as don't, while excluding numbers, punctuation-only tokens, and non-Latin strings. The same preprocessing is used for BPE and WordPiece, so the comparison is based on the same word-level input.

### Train a Naive BPE Tokenizer

```
bpe = NaiveBPE()
bpe.train(vocab_size=500, word_freqs=word_freqs)
```

### Train a Fast BPE Tokenizer

```
bpe = FastBPE()
bpe.train(vocab_size=1000, word_freqs=word_freqs)
```

### Train a WordPiece Tokenizer

The main WordPiece script is:
```
python wordpiece/run_wordpiece_experiments.py --help
```

To train the exact baseline WordPiece tokenizer:
```
python wordpiece/run_wordpiece_experiments.py \
  --training_method <baseline_or_fast> \
  --train_path <path_to_training_corpus> \
  --valid_path <path_to_validation_corpus> \
  --test_path <path_to_test_corpus> \
  --vocab_size <target_vocab_size> \
  --min_pair_freq <minimum_pair_frequency> \
  --max_eval_word_tokens <number_of_eval_word_tokens> \
  --output_dir <output_directory>
```

### Compare Tokenization Strategies

`python compare_tokenizers.py`

### Run a Single BPE Module

```
python bpe/bpe_fast.py
python bpe/bpe_naive.py
```

...

### Run evaluation

```
./eval/run_eval.sh
```
