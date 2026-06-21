import os
import pickle
import re
import tempfile
from collections import Counter

import morfessor
from datasets import load_dataset


class MorfessorModel:
    def __init__(self, corpusweight=1.5):
        self.io = morfessor.MorfessorIO()
        self.model = morfessor.BaselineModel(corpusweight=1.5)

    def train(self, base_name, train_set, save_path):
        trainset = load_dataset(base_name, train_set, split="train")
        counter = Counter()
        for e in trainset:
            text = e["text"].lower()
            words = re.findall(r"\w+\S*", text)
            counter.update(words)
        print(f"Vocabulary size: {len(counter)}")

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            for word, count in counter.items():
                tmp.write(f"{count} {word}\n")
            tmp_path = tmp.name
        try:
            train_data = list(self.io.read_corpus_file(tmp_path))
        finally:
            os.unlink(tmp_path)

        self.model.load_data(train_data)
        self.model.train_batch()

        if save_path:
            self.io.write_binary_model_file(save_path, self.model)
            print(f"Model saved to: {save_path}")
        return self.model

    def load(self, model_path):
        self.model = self.io.read_binary_model_file(model_path)
        return self.model

    def segment(self, word):
        segment, _ = self.model.viterbi_segment(word)
        return segment


def test():
    mo = MorfessorModel()
    mo.train("Salesforce/wikitext", "wikitext-103-v1", "../data/morf_wiki_103.bin")
    mo.load("../data/morf_wiki_103.bin")
    print(mo.segment("disrespectful"))
