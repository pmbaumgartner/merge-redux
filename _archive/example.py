from typing import List

from merge_redux import run
from merge_redux.core_nogap import run as run_nogap


if __name__ == "__main__":
    from itertools import chain
    from pathlib import Path

    corpus: List[List[str]] = []
    for txt_file in Path("tests/sample_corpus/").glob("*.TXT"):
        for line in txt_file.read_text().split("\n"):
            if line:
                tokens: List[str] = line.split(" ")
                corpus.append(tokens)

    initial_corpus_size = len(list(chain.from_iterable(corpus)))
    print(f"Lines (turns): {len(corpus):,}")
    print(f"Tokens (corpus_size in code): {initial_corpus_size:,}")

    winners = run(corpus, 0, 1000, min_bigram_freq=0, min_lexeme_freq=0)
    # print(*run(corpus, 0, 50, min_bigram_freq=0, min_lexeme_freq=0), sep="\n")
    # print(*run_nogap(corpus, 10, min_bigram_freq=0, min_lexeme_freq=0), sep="\n")
