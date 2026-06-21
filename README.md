### install from requirements.txt

`pip install -r requirements.txt`

### load data

填写

### normalization

填写

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
