from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data"

gtn = load_dataset("manu/project_gutenberg", split="en[-1%:]")
for e in gtn:
    if len(e["text"]) <= 100:
        continue
    text_gtn = e["text"]


with open(DATA_DIR / "guten_test.txt", "w") as f:
    f.write(text_gtn)


wk = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test[-1%:]")
for e in wk:
    if len(e["text"]) <= 100:
        continue
    text_wk = e["text"]

with open(DATA_DIR / "wiki_test.txt", "w") as f:
    f.write(text_wk)
