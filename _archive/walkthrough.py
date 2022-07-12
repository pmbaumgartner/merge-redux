# %% [markdown]
# # MERGE Algorithm Walkthrough
# This is a reimplementation of the MERGE algorithm. My hope is that by proceeding through the algorithm in a linear fashion, the order of operations will be clearer and will facilitate a refactoring into more generalized code.

# %% [markdown]
# ## Initialization
#
# When the model is initialized, it needs three things: a pre-processed corpus, a gapsize, and the number of iterations. We'll use the corpus included in the original implementation, which is "[...] a combination of the Santa Barbara Corpus of Spoken American English and the spoken component of the ICE Canada corpus"
#
# > Du Bois, John W., Wallace L. Chafe, Charles Meyers, Sandra A. Thompson, Nii Martey, and Robert Englebretson (2005). Santa Barbara corpus of spoken American English. Philadelphia: Linguistic Data Consortium.
#
# > Newman, John and Georgie Columbus (2010). The International Corpus of English – Canada. Edmonton, Alberta: University of Alberta.
#
#
# Note that these are dialog corpuses, so each line is often referred to as a `turn`. They've also been pre-processed with lowercasing, and punctuation replaced with alphanumeric substitutions (`_` --> `undrscr`).
#

from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Counter as CounterType
from typing import DefaultDict, Dict, List, NamedTuple, NewType, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# %%

# The original implementation had multiple delimiters,
# space delimiting gives the same n tokens so we'll simplify

# corpus: List[List[str]] = []
# for txt_file in Path("sample_corpus/").glob("*.TXT"):
#     for line in txt_file.read_text().split("\n"):
#         if line:
#             tokens: List[str] = line.split(" ")
#             corpus.append(tokens)

# initial_corpus_size = len(list(chain.from_iterable(corpus)))
# print(f"Lines (turns): {len(corpus):,}")
# print(f"Tokens (corpus_size in code): {initial_corpus_size:,}")


# %%
# Then store the tokens in the special data structure


# %%
# We create some simple data structures

_SMALL = 1e-10


class Word(NamedTuple):
    wordstr: str
    position: int

    def __repr__(self):
        return f"Word('{self.wordstr}', {self.position})"


# Lexeme can be used to store n-grams as the algorithm works
# plus we need to hash it so we use an expanding tuple


class Lexeme(NamedTuple):
    word: Tuple[Word, ...]
    token_index: int

    def __repr__(self):
        return f"Lexeme({self.word}, {self.token_index})"


LineIndex = NewType("LineIndex", int)
TokenIndex = NewType("TokenIndex", int)


@dataclass
class LexemeData:
    lexemes_to_locations: DefaultDict[
        Lexeme, Set[Tuple[LineIndex, TokenIndex]]
    ] = field(default_factory=lambda: defaultdict(set))
    locations_to_lexemes: DefaultDict[LineIndex, Dict[TokenIndex, Lexeme]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    locations_to_locations: Dict[
        Tuple[LineIndex, TokenIndex], Tuple[LineIndex, TokenIndex]
    ] = field(default_factory=dict)

    # NOTE: This is a Counter in original code, but doesn't use counter methods.
    # and is typed as a regular dict here
    lexemes_to_freqs: Dict[Lexeme, int] = field(default_factory=dict)
    turn_lengths: Dict[LineIndex, int] = field(default_factory=dict)

    def get_lexeme(
        self, turn_index: LineIndex, word_index: TokenIndex
    ) -> Tuple[Lexeme, TokenIndex]:
        """Gets the lexeme at a turn_index, word_index. This looks up the relevant
        Lexeme.token_index for the specificed index and then subtracts that from the given
        word_index to ensure you get the left anchor of any multi-word lexemes.

        Args:
            turn_index (LineIndex): The turn or line index.
            word_index (TokenIndex): The word index.

        Returns:
            Tuple[Lexeme, TokenIndex]: The left anchor lexeme and position of that anchor.
        """
        lexeme = self.locations_to_lexemes[turn_index][word_index]
        pos = TokenIndex(word_index - lexeme.token_index)
        left_lexeme = self.locations_to_lexemes[turn_index][pos]
        return left_lexeme, pos

    @classmethod
    def from_corpus(cls, corpus: List[List[str]]) -> "LexemeData":
        lexeme_data = cls()
        corpus_iter_progress = tqdm(
            enumerate(corpus),
            desc="Creating LexemeData from Corpus",
            unit="line",
            total=len(corpus),
        )
        for (line_ix, tokens) in corpus_iter_progress:
            for (word_ix, word) in enumerate(tokens):
                line_ix = LineIndex(line_ix)
                word_ix = TokenIndex(word_ix)
                lexeme = Lexeme(word=(Word(word, 0),), token_index=0)
                loc = (line_ix, word_ix)
                lexeme_data.lexemes_to_locations[lexeme].add(loc)
                lexeme_data.locations_to_lexemes[line_ix][word_ix] = lexeme
                lexeme_data.locations_to_locations[
                    loc
                ] = loc  # from original code, not sure what this does
        lexeme_data.lexemes_to_freqs = {
            k: len(v) for k, v in lexeme_data.lexemes_to_locations.items()
        }
        lexeme_data.turn_lengths = {
            LineIndex(line_ix): max(token_index) + 1
            for (line_ix, token_index) in lexeme_data.locations_to_lexemes.items()
        }
        return lexeme_data

    @property
    def corpus_size(self) -> int:
        """The total number of Lexemes within the corpus. Will get smaller as
        unigrams get merged into bigrams (as those are a single 'token')"""
        return sum(self.lexemes_to_freqs.values())


# Lexemes.count_frequencies()


# %%
# initial_lexemes = LexemeData.from_corpus(corpus)
# assert initial_corpus_size == initial_lexemes.corpus_size

# # We could double check this as a counter now
# Counter(initial_lexemes.lexemes_to_freqs).most_common(5)


# %% [markdown]
# # Bigram Data
# Bigram data is stored in another object called `Bigrams`.
# There is a lot of logic in here we are going to separate out.
#
# We also need to declare a `gapsize` at this point, which allows for MWEs with discontinuity (e.g. `in _ of`).

# %%
Gapsize = NewType("Gapsize", int)

# gapsize = Gapsize(1)


# %%
# Data structures


class Bigram(NamedTuple):
    el1: Lexeme
    el2: Lexeme
    gapsize: Gapsize


@dataclass
class BigramData:
    bigrams_to_freqs: CounterType[Bigram] = field(default_factory=Counter)
    bigrams_to_locations: Dict[Bigram, Set[Tuple[LineIndex, TokenIndex]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    left_lex_to_bigrams: Dict[Tuple[Lexeme, Gapsize], Set[Bigram]] = field(
        default_factory=lambda: defaultdict(set)
    )
    right_lex_to_bigrams: Dict[Tuple[Lexeme, Gapsize], Set[Bigram]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @classmethod
    def from_lexemes(cls, lexeme_data: LexemeData, gapsize: Gapsize) -> "BigramData":
        bigram_data = cls()
        corpus_iter_progress = tqdm(
            lexeme_data.turn_lengths.items(),
            desc="Creating BigramData from LexemeData",
            unit="line",
            total=len(lexeme_data.turn_lengths),
        )
        for line_ix, line_length in corpus_iter_progress:
            for curr_gapsize in range(gapsize + 1):
                # `rightmost_leftedge` in code
                # last token that can be part of a bigram. Think if gapsize = 0,
                # the the penultimate token will be the first element of
                # the final bigram
                # e.g. (a, b, c, d) w/ gapsize 1
                # last = 4 - 1 - 1 = 2
                # Then note we input this into range(last) which is not inclusive,
                # so this maps to indices 0, 1
                last_token_index = TokenIndex(line_length - curr_gapsize - 1)
                curr_gapsize = Gapsize(curr_gapsize)
                for ix in range(last_token_index):
                    ix = TokenIndex(ix)
                    right_ix = TokenIndex(ix + curr_gapsize + 1)
                    _left = lexeme_data.locations_to_lexemes[line_ix][ix]
                    _right = lexeme_data.locations_to_lexemes[line_ix][right_ix]
                    _location = lexeme_data.locations_to_locations.get(
                        (line_ix, ix), (line_ix, ix)
                    )
                    bgr = Bigram(el1=_left, el2=_right, gapsize=curr_gapsize)
                    bigram_data.add_bigram(bgr, _location)
        return bigram_data

    def add_bigram(
        self, bigram: Bigram, location: Tuple[LineIndex, TokenIndex]
    ) -> None:
        # Original code has a conditional here to only do this if the bigram
        # isn't in the frequency counter yet, but that's a lot of checks
        self.left_lex_to_bigrams[(bigram.el1, bigram.gapsize)].add(bigram)
        self.right_lex_to_bigrams[(bigram.el2, bigram.gapsize)].add(bigram)
        self.bigrams_to_freqs[bigram] += 1
        self.bigrams_to_locations[bigram].add(location)

    @property
    def type_count(self) -> int:
        return len(self.bigrams_to_freqs)

    def remove_bigram(self, bigram: Bigram):
        self.bigrams_to_freqs.pop(bigram)
        self.bigrams_to_locations.pop(bigram)
        self.left_lex_to_bigrams[(bigram.el1, bigram.gapsize)].remove(bigram)
        self.right_lex_to_bigrams[(bigram.el2, bigram.gapsize)].remove(bigram)


# initial_bigrams = BigramData.from_lexemes(initial_lexemes, gapsize)

# %% [markdown]
# ## Candidate Table
# The candidate table is a pandas dataframe that is used to store frequency information.
# In the original implementation it's something like a paged data frame with multiple `tables`.
# I don't see the reason for this so we'll simplify the implementation into a single dataframe.

# %%


def create_bigram_table(
    lexeme_data: LexemeData, bigram_data: BigramData
) -> pd.DataFrame:
    data = []
    for (bgr, freq) in bigram_data.bigrams_to_freqs.items():
        # Look up the frequency of each element of the bigrams
        el1_freq = lexeme_data.lexemes_to_freqs[bgr.el1]
        el2_freq = lexeme_data.lexemes_to_freqs[bgr.el2]
        row = {"bgr": bgr, "bgr_freq": freq, "el1_freq": el1_freq, "el2_freq": el2_freq}
        data.append(row)

    table = pd.DataFrame(data).set_index("bgr")
    return table


# table = create_bigram_table(initial_bigrams)
# table_og = table.copy()


# %% [markdown]
# ## Iterating the algorithm
#
# Now the fun begins. The first thing we do is calculate the 'winner' from the bigram
# tables by calculating the log-likelihood for each bigram.
# The bigram with the highest log-likelihood is the winner, and will then be merged.

# %%
# TODO: Validate this specific implementation of log-likelihood


def calculate_log_likelihood(
    statistics: pd.DataFrame, lexeme_data: LexemeData
) -> pd.Series:

    corpus_size = lexeme_data.corpus_size

    obsA = statistics["bgr_freq"]
    obsB = statistics["el1_freq"] - statistics["bgr_freq"]
    obsC = statistics["el2_freq"] - statistics["bgr_freq"]
    obsD = corpus_size - (obsA + obsB + obsC)

    expA = statistics["el2_freq"] / corpus_size * statistics["el1_freq"]
    expB = (corpus_size - statistics["el2_freq"]) / corpus_size * statistics["el1_freq"]
    expC = statistics["el2_freq"] / corpus_size * (corpus_size - statistics["el1_freq"])
    expD = (
        (corpus_size - statistics["el2_freq"])
        / corpus_size
        * (corpus_size - statistics["el1_freq"])
    )

    llA = obsA * np.log(obsA / (expA + _SMALL))
    llB = np.where(obsB != 0, obsB * np.log(obsB / (expB + _SMALL)), 0)
    llC = np.where(obsC != 0, obsC * np.log(obsC / (expC + _SMALL)), 0)
    llD = obsD * np.log(obsD / (expD + _SMALL))

    log_likelihood = 2 * (llA + llB + llC + llD)
    log_likelihood = np.where(llA > 0, log_likelihood, log_likelihood * -1)
    log_likelihood_series = pd.Series(log_likelihood, index=statistics.index)
    return log_likelihood_series


def calculate_winner(initial_lexemes: LexemeData, statistics: pd.DataFrame) -> Bigram:
    log_likelihood = calculate_log_likelihood(statistics, initial_lexemes)
    winner: Bigram = log_likelihood.idxmax()  # type: ignore
    return winner


SatellitePosition = NewType("SatellitePosition", int)
"""A satellite position is an integer representing the position of a word
relative to the leftmost element of a Lexeme.

e.g. Lexeme(('in', 0), ('of', 2))) has satellite positions [0, 2]. These are usually added to
a TokenIndex in forming a ContextPosition.."""

ContextPosition = NewType("ContextPosition", int)
"""A context position is a TokenIndex ± (SatellitePosition + Gapsize + 1).
It is used to identify candidate bigrams."""


@dataclass
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    bigram_locations: Set[Tuple[LineIndex, TokenIndex]]
    merge_token_count: int = 0
    merged_satellite_positions: Dict[
        Tuple[LineIndex, SatellitePosition], TokenIndex
    ] = field(default_factory=dict)

    @classmethod
    def from_bigram_with_data(cls, bigram: Bigram, bigram_data: BigramData):
        el1_words = list(bigram.el1.word)
        el2_words_repositioned = [
            Word(
                wordstr=el2_word.wordstr,
                position=(el2_word.position + bigram.gapsize + 1),
            )
            for el2_word in bigram.el2.word
        ]
        all_words = sorted(el1_words + el2_words_repositioned, key=lambda word: word[1])
        new_lexeme = Lexeme(word=tuple(all_words), token_index=0)

        locations = bigram_data.bigrams_to_locations[bigram]
        return cls(bigram, new_lexeme, locations)

    @property
    def satellite_lexemes(self) -> Dict[SatellitePosition, Lexeme]:
        """Generates satellite lexemes. This is used on the winning bigram lexeme,
        Lexeme(words=(el1, el2)) to convert it into a Dict of lexemes like:
        `{0: Lexeme((el1, el2), 0), 1: Lexeme((el1, el2)), 1)}`
        for each word in the winning lexeme.

        Keep in mind with gapsize > 0, you can get something like:
        `{0: Lexeme((el1, el2), 0), 2: Lexeme((el1, el2)), 2)}`

        Args:
            merge_token (Lexeme): The winning lexeme (formed from two prior Lexeme elements)

        Returns:
            Dict[int, Lexeme]: Satellite Lexemes indexed by token index.
        """
        satellite_lexemes = {0: self.merged_lexeme}
        for word in self.merged_lexeme.word:
            if word.position > 0:
                new_lexeme = Lexeme(self.merged_lexeme.word, word.position)
                satellite_lexemes[word.position] = new_lexeme
        return {SatellitePosition(k): v for k, v in satellite_lexemes.items()}

    def satellite_positions(self, token_index: TokenIndex) -> List[SatellitePosition]:
        return [
            SatellitePosition(token_index + word.position)
            for word in self.merged_lexeme.word
        ]

    def generate_context_positions(
        self, turn_length: int, token_index: TokenIndex, gapsize: Gapsize
    ) -> List[Tuple[ContextPosition, SatellitePosition]]:
        """Generates context positions from the merged lexeme.
        For a given token_index and gapsize a context position is:

        TokenIndex ± (SatellitePosition + Gapsize + 1)

        e.g. for the first token, with a merged token position of 0, and no gap, it's the immediate
        right token: 0 + 0 + 0 + 1 = 1

        e.g. for the fifth token, with a merged word position of 2, gapsize 1 it's:
        5 + 2 + 1 + 1 = 9 (from a discontiuous bigram with a discontinuous satellite from the gap)

        It also calculates it to the left, so convert the operations to subtraction for that.

        Args:
            merge_token (Lexeme): The merge token to use word positions from.
            turn_length (int): The length of the turn (used to filter satellite positions)
            token_index (TokenIndex): The reference token index to generate positions from.
            gapsize: (Gapsize): The gapsize to generate satellites for.

        Returns:
            _type_: _description_
        """
        satellite_positions: List[SatellitePosition] = [
            SatellitePosition(token_index + word.position)
            for word in self.merged_lexeme.word
        ]
        context_positions: List[Tuple[ContextPosition, SatellitePosition]] = []
        for satellite_position in satellite_positions:
            for curr_gapsize in range(gapsize + 1):
                curr_gapsize = Gapsize(curr_gapsize)
                left_context_position = ContextPosition(
                    satellite_position - curr_gapsize - 1
                )
                if left_context_position >= 0:
                    curr_contextpos = (left_context_position, satellite_position)
                    context_positions.append(curr_contextpos)
                right_context_position = ContextPosition(
                    satellite_position + curr_gapsize + 1
                )
                if right_context_position < turn_length:
                    curr_contextpos = (
                        right_context_position,
                        satellite_position,
                    )
                    context_positions.append(curr_contextpos)
        return context_positions


# %%
# This was checked against the implementation + was correct.
# (((Word('you', 0),), 0), ((Word('know', 0),), 0), 0)         28104.733018

# %%
# winner_info = calculate_winner(initial_lexemes, table)

# %% [markdown]


# ## Merging
# We now create the merged token, which will then be substituted into the original corpus where it occured.

# %%


# merge_token = merge_bigram_to_lexeme(winner_info)
# merge_tracker: List[Lexeme] = []  # This is a Dict[iter, Lexeme] in the source
# merge_tracker.append(merge_token)


# %% [markdown]
# ## Bigram Updater
#
# This is the most complex part of the algorithm. The steps of the bigram updater are:
#
# - `Lexemes.set_merge_token` - creates `satellite_lexemes` a dict of position
#  to new `Lexeme` with `token_index`=`position`
#
#

# %%
# MAIN PART OF THE ALGORITHM STARTS HERE
#


# %%
# iterate through winner locations for this next part


# winner_locations = find_bigram_locations(winner_info, initial_bigrams)


def calculate_new_and_conflicting_bigrams(
    winner: WinnerInfo,
    lexeme_data: LexemeData,
    gapsize: Gapsize,
) -> Tuple[BigramData, BigramData]:
    # TODO: Can we separate this logic from the merged satellite positions?
    merge_token_count = 0
    new_bigrams = BigramData()
    conflicting_bigrams = BigramData()
    merged_satellite_positions: Dict[
        Tuple[LineIndex, SatellitePosition], TokenIndex
    ] = {}

    for (line_ix, word_ix) in winner.bigram_locations:
        # Check if it's among the existing conflicting bigrams
        # the first iteration we wont have this, so we'll come back to it
        if (line_ix, word_ix) in conflicting_bigrams.bigrams_to_locations[
            winner.bigram
        ]:
            # BigramUpdater.bigram_is_among_existing_conflicting_bigrams
            continue

        merge_token_count += 1

        # context_positions = BigramUpdater.context_pos_manager.generate_positions_around_satellites
        curr_turn_length = lexeme_data.turn_lengths[line_ix]

        context_positions = winner.generate_context_positions(
            curr_turn_length, word_ix, gapsize
        )
        # BigramUpdater.vicinity_lex_manager.create_bigrams_with_lexemes_surrounding_satellites(
        # merged_satellite_positions is an empty dict on iter = 0 but referenced here
        for context_position_info in context_positions:
            # Context_position_info gets unpacked in original code if you're looking for those variables
            context_pos, satellite_position = context_position_info
            if context_pos in winner.satellite_positions(word_ix):
                # TODO: Investigate what this does, without it a bunch
                # of token counts are added
                continue

            premerge_lexeme, premerge_leftanchor = lexeme_data.get_lexeme(
                line_ix, TokenIndex(satellite_position)
            )
            context_loc = (line_ix, TokenIndex(context_pos))

            # Original Comment:
            # Don't need to create conflicting bigram since this confl
            # same bigram would have already been created when the adjacent
            # merge token was created at an earlier iteration
            if context_loc in merged_satellite_positions:
                context_lexeme = winner.merged_lexeme
                context_ix = merged_satellite_positions[
                    (context_loc[0], SatellitePosition(context_loc[1]))
                ]
            else:
                context_lexeme, context_ix = lexeme_data.get_lexeme(
                    line_ix, TokenIndex(context_pos)
                )

            # BigramUpdater.create_new_bigram()
            # new_bigrams are those where el1 or el2 are now merge_token
            left_context = context_ix < word_ix
            if left_context:
                # all_lexemes.get_extant_loc_object(turn_number, context_ix )
                location_tuple = lexeme_data.locations_to_locations.get(
                    (line_ix, context_ix), (line_ix, context_ix)
                )
                gap_between_anchors = Gapsize(word_ix - context_ix - 1)
                _left, _right = context_lexeme, winner.merged_lexeme
            else:  # right context
                location_tuple = lexeme_data.locations_to_locations.get(
                    (line_ix, word_ix), (line_ix, word_ix)
                )
                # order reversed from left_context
                gap_between_anchors = Gapsize(context_ix - word_ix - 1)
                _left, _right = winner.merged_lexeme, context_lexeme
            # reimplemtation of Bigrams.save_bigram_data
            bgr = Bigram(el1=_left, el2=_right, gapsize=gap_between_anchors)
            new_bigrams.add_bigram(bgr, location_tuple)

            # BigramUpdater.create_conflicting_bigram()
            # conflicting_bigrams are those where pre-merge formed
            # a bigram with the left anchor of the new bigram.
            # e.g. merge_token = (c, d)
            # span (a, b, c, d, e)
            # (b, c) is a conflicting bigram (left context)
            # beacuse the new bigram is (b, (c, d))

            left_context = context_ix < premerge_leftanchor
            if left_context:
                # all_lexemes.get_extant_loc_object(turn_number, context_ix )
                location_tuple = lexeme_data.locations_to_locations.get(
                    (line_ix, context_ix), (line_ix, context_ix)
                )
                gap_between_anchors = Gapsize(premerge_leftanchor - context_ix - 1)
                _left, _right = context_lexeme, premerge_lexeme
            else:  # right context
                location_tuple = lexeme_data.locations_to_locations.get(
                    (line_ix, premerge_leftanchor), (line_ix, premerge_leftanchor)
                )
                # order reversed from left_context
                gap_between_anchors = Gapsize(context_ix - premerge_leftanchor - 1)
                _left, _right = premerge_lexeme, context_lexeme
            bgr = Bigram(el1=_left, el2=_right, gapsize=gap_between_anchors)
            conflicting_bigrams.add_bigram(bgr, location_tuple)

        for satellite_position in winner.satellite_positions(word_ix):
            merged_satellite_positions[(line_ix, satellite_position)] = word_ix
    winner.merge_token_count = merge_token_count
    winner.merged_satellite_positions = merged_satellite_positions

    return (new_bigrams, conflicting_bigrams)


# (
#     new_bigrams,
#     conflicting_bigrams,
#     merged_satellite_positions,
#     merge_token_count,
# ) = calculate_new_and_conflicting_bigrams(winner_locations, initial_lexemes)
# %%
# type_count_check = (new_bigrams.type_count, conflicting_bigrams.type_count)
# original_impl = (3531, 5207)
# print(
#     f"new_bigrams_diff: {original_impl[0]-type_count_check[0]}, "
#     f"conflicting_bigrams_diff: {original_impl[1]-type_count_check[1]}"
# )


# %%
# BigramUpdater.update_all_lexemes_with_merge_tokens()
# (after main_control_loop)
def update_all_lexemes_with_merge_tokens(
    winner_info: WinnerInfo,
    lexeme_data: LexemeData,
):
    lexeme_data.lexemes_to_freqs[
        winner_info.merged_lexeme
    ] = winner_info.merge_token_count

    for (
        line_ix,
        satellite_pos,
    ), word_ix in winner_info.merged_satellite_positions.items():
        # all_lexemes.add_merge_token(turn, satellite_pos, merge_token_leftanchor)
        # This basically just updates the `token_index` value on Lexeme for the new
        # bigram formed lexeme. The first satellite_pos == 0.
        loc = lexeme_data.locations_to_locations[(line_ix, TokenIndex(satellite_pos))]
        satellite_lexeme = winner_info.satellite_lexemes[
            SatellitePosition(satellite_pos - word_ix)
        ]
        lexeme_data.lexemes_to_locations[satellite_lexeme].add(loc)
        lexeme_data.locations_to_lexemes[line_ix][
            TokenIndex(satellite_pos)
        ] = satellite_lexeme


# update_all_lexemes_with_merge_tokens(
#     merge_token, initial_lexemes, merged_satellite_positions
# )
# %% [markdown]
# ## Frequency updater
#
# Now we handle the frequency updater, which we use for handling the statistics and log log_likelihood.
#
# ```python
# self.frequency_updater.set_new_freqs_for_elements_in_winner(
#     self.all_lexemes, self.winner_info, self.merge_token_count
# )
# self.frequency_updater.process_bgrs_with_same_element_types_as_winner(
#     self.gapsize, self.all_tables, self.all_bigrams
# )
# self.frequency_updater.add_new_bigrams(self.new_bigrams)
# self.frequency_updater.update_conflicting_bigram_freqs(
#     self.conflicting_bigrams
# )
# self.frequency_updater.remove_winner()
# ```

# %%


# %%
# subtract total count from each element
# FrequencyUpdater.set_new_freqs_for_elements_in_winner
# NOTE: Running this several times will continually subtract
def update_lexeme_counts_with_merged_elements(
    winner_info: WinnerInfo, lexeme_data: LexemeData
):
    el1_freq = lexeme_data.lexemes_to_freqs[winner_info.bigram.el1]
    new_el1_freq = el1_freq - winner_info.merge_token_count
    lexeme_data.lexemes_to_freqs[winner_info.bigram.el1] = new_el1_freq

    el2_freq = lexeme_data.lexemes_to_freqs[winner_info.bigram.el2]
    new_el2_freq = el2_freq - winner_info.merge_token_count
    lexeme_data.lexemes_to_freqs[winner_info.bigram.el2] = new_el2_freq


# update_lexeme_counts_with_merged_elements(
#     winner_info, merge_token_count, initial_lexemes
# )

# new_corpus_size = initial_corpus_size - merge_token_count
# assert (
#     initial_lexemes.corpus_size
#     == new_corpus_size
#     == initial_corpus_size - merge_token_count
# ), (
#     initial_corpus_size,
#     initial_lexemes.corpus_size,
#     new_corpus_size,
# )

# FrequencyUpdater.process_bgrs_with_same_element_types_as_winner
# get bigrams with elements of winner


def update_bigram_statistics_with_merged_elements(
    winner_info: WinnerInfo,
    gapsize: Gapsize,
    bigram_data: BigramData,
    lexeme_data: LexemeData,
    statistics: pd.DataFrame,
):
    # MUST BE RUN AFTER LEXEME COUNTS IS UPDATED!
    new_el1_freq = lexeme_data.lexemes_to_freqs[winner_info.bigram.el1]
    new_el2_freq = lexeme_data.lexemes_to_freqs[winner_info.bigram.el2]

    for curr_gapsize in range(gapsize + 1):
        curr_gapsize = Gapsize(curr_gapsize)
        el1_left_pos = bigram_data.left_lex_to_bigrams[
            (winner_info.bigram.el1, curr_gapsize)
        ]
        el1_right_pos = bigram_data.right_lex_to_bigrams[
            (winner_info.bigram.el1, curr_gapsize)
        ]
        el2_left_pos = bigram_data.left_lex_to_bigrams[
            (winner_info.bigram.el2, curr_gapsize)
        ]
        el2_right_pos = bigram_data.right_lex_to_bigrams[
            (winner_info.bigram.el2, curr_gapsize)
        ]

        # Note
        bigram_pos_columns = [
            (el1_left_pos, "el1_freq"),
            (el2_left_pos, "el2_freq"),
            (el1_right_pos, "el1_freq"),
            (el2_right_pos, "el2_freq"),
        ]
        for bigrams, col in bigram_pos_columns:
            for bigram in bigrams:
                value = new_el1_freq if col.startswith("el1") else new_el2_freq
                statistics.at[bigram, col] = value
        # This can definitely be optimized if we could just join these
        # rather than a lookup at each individual row


# %%
original_el1_freq, original_el2_freq = (13662, 2471)
# assert (new_el1_freq, new_el2_freq) == (original_el1_freq, original_el2_freq)


# %%
# FrequencyUpdater.add_new_bigrams

# Going to make a copy just in case we want to validate against the original
# collected_bigrams: BigramData = copy(initial_bigrams)

# Bigrams.add(new_bigrams)
# for all these loops: collected = initial (lookup) + new (in iter)


def update_bigram_data(bigram_data: BigramData, new_bigrams: BigramData):
    for bgr, freq in new_bigrams.bigrams_to_freqs.items():
        bigram_data.bigrams_to_freqs[bgr] += freq
        curr_locs = bigram_data.bigrams_to_locations[bgr]
        bigram_data.bigrams_to_locations[bgr] = curr_locs.union(
            new_bigrams.bigrams_to_locations[bgr]
        )

    for (el1, curr_gapsize), bigrams in new_bigrams.left_lex_to_bigrams.items():
        curr_left_lex_to_bigrams = bigram_data.left_lex_to_bigrams[(el1, curr_gapsize)]
        bigram_data.left_lex_to_bigrams[
            (el1, curr_gapsize)
        ] = curr_left_lex_to_bigrams.union(bigrams)

    for (el2, curr_gapsize), bigrams in new_bigrams.right_lex_to_bigrams.items():
        curr_right_lex_to_bigrams = bigram_data.right_lex_to_bigrams[
            (el2, curr_gapsize)
        ]
        bigram_data.right_lex_to_bigrams[
            (el2, curr_gapsize)
        ] = curr_right_lex_to_bigrams.union(bigrams)


# update_bigram_data(collected_bigrams, new_bigrams)


def update_bigram_statistics(
    statistics: pd.DataFrame,
    bigram_data: BigramData,
    lexeme_data: LexemeData,
    new_bigrams: BigramData,
):
    new_bigram_records = []
    for bigram in new_bigrams.bigrams_to_freqs:
        new_freq = bigram_data.bigrams_to_freqs[bigram]
        new_el1_freq = lexeme_data.lexemes_to_freqs[bigram.el1]
        new_el2_freq = lexeme_data.lexemes_to_freqs[bigram.el2]
        # FrequencyUpdater.new_data_for_tables.push_row
        # Which just adds all these to one list - I think I can just append

        # FrequencyUpdater.all_tables.add
        # This can be made much simpler since we arent doing the broken tables
        new_bigram_records.append(
            {
                "bgr": bigram,
                "bgr_freq": new_freq,
                "el1_freq": new_el1_freq,
                "el2_freq": new_el2_freq,
            }
        )

    new_bigram_df = pd.DataFrame(new_bigram_records).set_index("bgr")
    statistics = pd.concat((statistics, new_bigram_df), axis="index")


# update_bigram_statistics(table, collected_bigrams, initial_lexemes, new_bigrams)
# %%
# FrequencyUpdater.update_conflicting_bigram_freqs
# all_bigrams.deduct_freqs(self.conflicting_bigrams)
# Here all bigrams is collected_bigrams (which has been modified above)


def update_conflicting_bigrams(
    bigram_data: BigramData, conflicting_bigrams: BigramData
):
    for bgr, freq in conflicting_bigrams.bigrams_to_freqs.items():
        bigram_data.bigrams_to_freqs[bgr] -= freq
        curr_locs = conflicting_bigrams.bigrams_to_locations[bgr]
        for loc in curr_locs:
            bigram_data.bigrams_to_locations[bgr].remove(loc)
        if bigram_data.bigrams_to_freqs[bgr] < 1:
            del bigram_data.bigrams_to_freqs[bgr]
            del bigram_data.bigrams_to_locations[bgr]
            bigram_data.left_lex_to_bigrams[(bgr.el1, bgr.gapsize)].remove(bgr)
            bigram_data.right_lex_to_bigrams[(bgr.el2, bgr.gapsize)].remove(bgr)


# update_conflicting_bigrams(collected_bigrams, conflicting_bigrams)


def update_conflicting_bigram_statistics(
    statistics: pd.DataFrame, conflicting_bigrams: BigramData, bigram_data: BigramData
):
    for bigram in conflicting_bigrams.bigrams_to_freqs:
        statistics.at[bigram, "bgr_freq"] = bigram_data.bigrams_to_freqs[bigram]


def remove_winner_from_statistics(statistics: pd.DataFrame, winner_info: WinnerInfo):
    statistics.at[winner_info.bigram, "bgr_freq"] = 0


# update_conflicting_bigram_statistics(table, conflicting_bigrams)
# %%
# FrequencyUpdater.remove_winner()


def remove_winner_from_bigram_data(winner: Bigram, bigram_data: BigramData):
    bigram_data.bigrams_to_freqs.pop(winner)
    bigram_data.bigrams_to_locations.pop(winner)
    bigram_data.left_lex_to_bigrams[(winner.el1, winner.gapsize)].remove(winner)
    bigram_data.right_lex_to_bigrams[(winner.el2, winner.gapsize)].remove(winner)


# remove_winner_from_bigram_data(winner_info, collected_bigrams)
# 537076
# collected_bigrams.type_count

# 540413
# len(table)

# %% [markdown]
# **✨ That's an iteration!**

# %%
# print a winner
# for line, lexemes in initial_lexemes.locations_to_lexemes.items():  # type: ignore
#     if merge_token in lexemes.values():
#         print(line)
#         for i, lex in lexemes.items():
#             print(i, lex)
#         break

# %%
