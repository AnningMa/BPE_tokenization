import json
from pathlib import Path
from bpe.bpe_fast import FastBPE


def main(vocab_size: int = 1000, top_n: int = 20):
    data_path = Path(__file__).parent / "data" / "word_freqs_train_top5000.json"
    with open(data_path, encoding="utf-8") as f:
        word_freqs = json.load(f)

    bpe = FastBPE()
    bpe.train(vocab_size=vocab_size, word_freqs=word_freqs)

    ambiguous = []
    for word, freq in word_freqs.items():
        standard = bpe.tokenize(word)
        longest = bpe.tokenize_longest(word)
        if standard != longest:
            ambiguous.append({
                "word": word,
                "freq": freq,
                "tokenize": standard,
                "tokenize_longest": longest,
            })

    ambiguous.sort(key=lambda x: x["freq"], reverse=True)

    total = len(word_freqs)
    print(f"ambiguity: {len(ambiguous)} / {total} ({len(ambiguous)/total*100:.2f}%)")
    print(f"vocab_size: {len(bpe.vocab)}, merges: {len(bpe.merges)}\n")

    for item in ambiguous[:top_n]:
        print(f"{item['word']:20} freq={item['freq']:,}")
        print(f"{' '*20} tokenize:         {item['tokenize']}")
        print(f"{' '*20} tokenize_longest: {item['tokenize_longest']}")


if __name__ == "__main__":
    main()
