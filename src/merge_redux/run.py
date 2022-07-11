from typing import List

from tqdm import trange

from core import (
    Bigram,
    BigramData,
    Gapsize,
    LexemeData,
    WinnerInfo,
    calculate_new_and_conflicting_bigrams,
    calculate_winner,
    create_bigram_table,
    remove_winner_from_bigram_data,
    update_all_lexemes_with_merge_tokens,
    update_bigram_data,
    update_conflicting_bigrams,
    update_lexeme_counts_with_merged_elements,
)


if __name__ == "__main__":
    from itertools import chain
    from pathlib import Path

    corpus: List[List[str]] = []
    for txt_file in Path("sample_corpus/").glob("*.TXT"):
        for line in txt_file.read_text().split("\n"):
            if line:
                tokens: List[str] = line.split(" ")
                corpus.append(tokens)

    initial_corpus_size = len(list(chain.from_iterable(corpus)))
    print(f"Lines (turns): {len(corpus):,}")
    print(f"Tokens (corpus_size in code): {initial_corpus_size:,}")

    print(*run(corpus, 1, 10), sep="\n")

# Lexeme((Word('ladies', 0), Word('gentlemen', 2)), 0)
# not saving the correct gapsize
