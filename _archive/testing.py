from itertools import chain
from pathlib import Path
from typing import List

from tqdm import trange

from src.merge_redux.core import (
    Bigram,
    BigramData,
    Gapsize,
    Lexeme,
    LexemeData,
    WinnerInfo,
    calculate_new_and_conflicting_bigrams,
    calculate_winner,
    create_bigram_table,
    remove_winner_from_bigram_data,
    remove_winner_from_statistics,
    update_all_lexemes_with_merge_tokens,
    update_bigram_data,
    update_bigram_statistics,
    update_bigram_statistics_with_merged_elements,
    update_conflicting_bigram_statistics,
    update_conflicting_bigrams,
    update_lexeme_counts_with_merged_elements,
)

corpus: List[List[str]] = [["0", "1", "2", "0", "1", "3", "4", "0", "1"]]

# After 2 iters, winner: ["3" (0), "0" (1), "1" (3)]
corpus: List[List[str]] = [["5", "0", "2", "1", "3", "0", "3", "1", "4"]]
# Wrong


gapsize = 1

winners: List[Bigram] = []
gapsize = Gapsize(gapsize)
initial_lexemes = LexemeData.from_corpus(corpus)
initial_bigrams = BigramData.from_lexemes(initial_lexemes, gapsize)
initial_statistics = create_bigram_table(initial_lexemes, initial_bigrams)
for _ in trange(iters):
    # Winning stuff
    winner = calculate_winner(initial_lexemes, initial_statistics)
    winner_info = WinnerInfo.from_bigram_with_data(winner, initial_bigrams)
    winners.append(winner_info.bigram)
    # Cleanup stuff
    (
        new_bigrams,
        conflicting_bigrams,
    ) = calculate_new_and_conflicting_bigrams(winner_info, initial_lexemes, gapsize)
    # Mutating in-place now!
    update_all_lexemes_with_merge_tokens(winner_info, initial_lexemes)
    update_lexeme_counts_with_merged_elements(winner_info, initial_lexemes)
    # update_bigram_statistics_with_merged_elements(
    #     winner_info, gapsize, initial_bigrams, initial_lexemes, initial_statistics
    # )
    update_bigram_data(initial_bigrams, new_bigrams)
    # update_bigram_statistics(
    #     initial_statistics, initial_bigrams, initial_lexemes, new_bigrams
    # )
    update_conflicting_bigrams(initial_bigrams, conflicting_bigrams)
    # update_conflicting_bigram_statistics(
    #     initial_statistics, conflicting_bigrams, initial_bigrams
    # )
    # remove_winner_from_statistics(initial_statistics, winner_info)
    remove_winner_from_bigram_data(winner, initial_bigrams)
    initial_statistics = create_bigram_table(initial_lexemes, initial_bigrams)


# if __name__ == "__main__":

#     print(*run(corpus, 1, 250), sep="\n")

# Lexeme((Word('ladies', 0), Word('gentlemen', 2)), 0)
# not saving the correct gapsize
