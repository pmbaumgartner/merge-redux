from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Counter as CounterType
from typing import DefaultDict, Dict, List, NamedTuple, NewType, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm, trange


_SMALL = 1e-10


class Word(NamedTuple):
    wordstr: str
    position: int

    def __repr__(self):
        return f"Word('{self.wordstr}', {self.position})"


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


Gapsize = NewType("Gapsize", int)


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
        # isn't in the frequency counter yet, but that's a lot of checks that `add` can
        # do for us anyway
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
        if (line_ix, word_ix) in conflicting_bigrams.bigrams_to_locations[
            winner.bigram
        ]:
            # TODO: Do we need this?
            continue

        merge_token_count += 1

        curr_turn_length = lexeme_data.turn_lengths[line_ix]

        context_positions = winner.generate_context_positions(
            curr_turn_length, word_ix, gapsize
        )
        for context_position_info in context_positions:
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
            bgr = Bigram(el1=_left, el2=_right, gapsize=gap_between_anchors)
            new_bigrams.add_bigram(bgr, location_tuple)

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
        loc = lexeme_data.locations_to_locations[(line_ix, TokenIndex(satellite_pos))]
        satellite_lexeme = winner_info.satellite_lexemes[
            SatellitePosition(satellite_pos - word_ix)
        ]
        lexeme_data.lexemes_to_locations[satellite_lexeme].add(loc)
        lexeme_data.locations_to_lexemes[line_ix][
            TokenIndex(satellite_pos)
        ] = satellite_lexeme


def update_lexeme_counts_with_merged_elements(
    winner_info: WinnerInfo, lexeme_data: LexemeData
):
    el1_freq = lexeme_data.lexemes_to_freqs[winner_info.bigram.el1]
    new_el1_freq = el1_freq - winner_info.merge_token_count
    lexeme_data.lexemes_to_freqs[winner_info.bigram.el1] = new_el1_freq

    el2_freq = lexeme_data.lexemes_to_freqs[winner_info.bigram.el2]
    new_el2_freq = el2_freq - winner_info.merge_token_count
    lexeme_data.lexemes_to_freqs[winner_info.bigram.el2] = new_el2_freq


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


def update_conflicting_bigram_statistics(
    statistics: pd.DataFrame, conflicting_bigrams: BigramData, bigram_data: BigramData
):
    for bigram in conflicting_bigrams.bigrams_to_freqs:
        statistics.at[bigram, "bgr_freq"] = bigram_data.bigrams_to_freqs[bigram]


def remove_winner_from_statistics(statistics: pd.DataFrame, winner_info: WinnerInfo):
    statistics.at[winner_info.bigram, "bgr_freq"] = 0


def remove_winner_from_bigram_data(winner: Bigram, bigram_data: BigramData):
    bigram_data.bigrams_to_freqs.pop(winner)
    bigram_data.bigrams_to_locations.pop(winner)
    bigram_data.left_lex_to_bigrams[(winner.el1, winner.gapsize)].remove(winner)
    bigram_data.right_lex_to_bigrams[(winner.el2, winner.gapsize)].remove(winner)


def run(corpus: List[List[str]], gapsize: int, iters: int):
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
        # Mutating these in-place now!
        update_all_lexemes_with_merge_tokens(winner_info, initial_lexemes)
        update_lexeme_counts_with_merged_elements(winner_info, initial_lexemes)
        update_bigram_data(initial_bigrams, new_bigrams)
        update_conflicting_bigrams(initial_bigrams, conflicting_bigrams)
        remove_winner_from_bigram_data(winner, initial_bigrams)
        initial_statistics = create_bigram_table(initial_lexemes, initial_bigrams)

    return winners
