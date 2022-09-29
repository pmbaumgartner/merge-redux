from typing import List

from merge_redux.simple import run


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

    winners = run(corpus, 10, output=None)
