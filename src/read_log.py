import json

import pandas as pd

df = pd.read_csv("../log/tokenize-log.csv").rename(
    columns={
        "freq.n_preserved(0k)": "freq.n_preserved(1k)",
        "on_rare_words": "rare.avg_fertility",
    }
)


df[
    [
        "type",
        "train_data",
        "vocab_size",
        "min_pair_freq",
        "gold.precision",
        "gold.recall",
        "gold.f1",
        "freq.n_preserved(1k)",
        "freq.avg_fertility",
        "rare.avg_fertility",
    ]
].to_csv("../log/partical-log.csv")
