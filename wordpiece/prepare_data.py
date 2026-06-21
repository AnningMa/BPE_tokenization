import argparse
from pathlib import Path
from datasets import load_dataset

def write_text_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if line:
                f.write(line + "\n")

def prepare_wikitext(output_dir: Path) -> None:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    wikitext_dir = output_dir / "wikitext-103"
    
    write_text_file(wikitext_dir / "wiki.train.tokens", [row["text"] for row in dataset["train"]])
    write_text_file(wikitext_dir / "wiki.valid.tokens", [row["text"] for row in dataset["validation"]])
    write_text_file(wikitext_dir / "wiki.test.tokens", [row["text"] for row in dataset["test"]])
    print(f"WikiText-103 saved to {wikitext_dir}")

def prepare_gutenberg(output_dir: Path) -> None:
    dataset = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    gutenberg_dir = output_dir / "gutenberg"

    train_600_lines = []
    train_1200_lines = []

    for book_id, book in enumerate(dataset):
        if book_id >= 1200:
            break
        
        lines = [line.strip() for line in book["TEXT"].split("\n") if line.strip()]
        
        train_1200_lines.extend(lines)
        if book_id < 600:
            train_600_lines.extend(lines)

    write_text_file(gutenberg_dir / "gutenberg_600_train.txt", train_600_lines)
    write_text_file(gutenberg_dir / "gutenberg_1200_train.txt", train_1200_lines)
    print(f"Gutenberg training files saved to {gutenberg_dir}")
    print("Gutenberg test file is already included as data/guten_test_chunk.txt")

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prepare_wikitext(output_dir)
    prepare_gutenberg(output_dir)

if __name__ == "__main__":
    main()