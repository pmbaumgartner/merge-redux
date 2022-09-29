from collections import Counter, defaultdict
from copy import copy
from dataclasses import field
from itertools import groupby
from typing import Any, Counter as CounterType, Iterable, Optional, Sized
from typing import DefaultDict, Dict, List, NamedTuple, NewType, Set, Tuple
from pydantic import BaseModel, NonNegativeInt, Field, ValidationError, PositiveInt
from pydantic.dataclasses import dataclass
from itertools import zip_longest

import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange

_SMALL = 1e-10


class Lexeme(BaseModel):
    word: Tuple[str, ...]
    index: NonNegativeInt

    def __repr__(self):
        return f"({self.word}, {self.index})"

    class Config:
        allow_mutation = False
        frozen = True


corpus = ["a a a a".split()]

corpus = ["c a b a b a b d".split()]


# init
# ((a,), 0), ((a,), 0), ((a,), 0), ((a,), 0)
# ((a,a), 0), ((a,a), 1), ((a,a), 0), ((a,a), 1)
# ((a,a,a,a), 0), ((a,a,a,a), 1), ((a,a,a,a), 2), ((a,a,a,a), 3)

LineIndex = NewType("LineIndex", NonNegativeInt)
TokenIndex = NewType("TokenIndex", NonNegativeInt)


@dataclass
class LexemeData:
    lexemes_to_locations: DefaultDict[
        Lexeme, Set[Tuple[LineIndex, TokenIndex]]
    ] = Field(default_factory=lambda: defaultdict(set))
    locations_to_lexemes: DefaultDict[LineIndex, Dict[TokenIndex, Lexeme]] = Field(
        default_factory=lambda: defaultdict(dict)
    )
    lexemes_to_freqs: Dict[Lexeme, PositiveInt] = Field(default_factory=dict)
    line_lengths: Dict[LineIndex, PositiveInt] = Field(default_factory=dict)

    def get_lexeme(
        self, line_index: LineIndex, word_index: TokenIndex
    ) -> Tuple[Lexeme, TokenIndex]:
        """Gets the lexeme at a line_index, word_index. This looks up the relevant
        Lexeme.token_index for the specificed index and then subtracts that from the given
        word_index to ensure you get the left anchor of any multi-word lexemes.

        Args:
            line_index (LineIndex): The line index.
            word_index (TokenIndex): The word index.

        Returns:
            Tuple[Lexeme, TokenIndex]: The left anchor lexeme and position of that anchor.
        """
        lexeme = self.locations_to_lexemes[line_index][word_index]
        if lexeme.index != 0:
            pos = TokenIndex(word_index - lexeme.index)
            lexeme = self.locations_to_lexemes[line_index][pos]
        else:
            pos = word_index
        return lexeme, pos

    def get(
        self, line_index: LineIndex, word_index: TokenIndex, *, default=None
    ) -> Optional[Lexeme]:
        """Get Lexeme but only lexeme"""
        lexeme = self.locations_to_lexemes[line_index].get(word_index, default)
        if lexeme is not None:
            if lexeme.index != 0:
                pos = TokenIndex(word_index - lexeme.index)
                lexeme = self.locations_to_lexemes[line_index].get(pos, default)
        return lexeme

    @classmethod
    def from_corpus(cls, corpus: Iterable[Iterable[str]]) -> "LexemeData":
        lexeme_data = cls()
        total: Optional[int] = len(corpus) if isinstance(corpus, Sized) else None
        corpus_iter_progress = tqdm(
            enumerate(corpus),
            desc="Creating LexemeData from Corpus",
            unit="line",
            total=total,
        )
        for (line_ix, tokens) in corpus_iter_progress:
            for (word_ix, word) in enumerate(tokens):
                line_ix = LineIndex(line_ix)
                word_ix = TokenIndex(word_ix)
                lexeme = Lexeme(word=(word,), index=0)
                loc = (line_ix, word_ix)
                lexeme_data.lexemes_to_locations[lexeme].add(loc)
                lexeme_data.locations_to_lexemes[line_ix][word_ix] = lexeme

        lexeme_data.lexemes_to_freqs = {
            k: len(v) for k, v in lexeme_data.lexemes_to_locations.items()
        }
        lexeme_data.line_lengths = {
            LineIndex(line_ix): max(token_index) + 1
            for (line_ix, token_index) in lexeme_data.locations_to_lexemes.items()
        }
        return lexeme_data

    @property
    def corpus_size(self) -> int:
        """The total number of Lexemes within the corpus. Will get smaller as
        unigrams get merged into bigrams (as those are a single 'token')"""
        return sum(self.lexemes_to_freqs.values())

    def render_corpus(self) -> List[List[Lexeme]]:
        corpus = []
        for line_ix in self.locations_to_lexemes:
            line_tokens = []
            for token_ix in self.locations_to_lexemes[line_ix]:
                line_tokens.append(self.locations_to_lexemes[line_ix][token_ix])
            corpus.append(line_tokens)
        return corpus

    def locations_to_root_lexemes(self, line: LineIndex):
        lexeme_dicts = self.locations_to_lexemes[line]
        return {k: v for k, v in lexeme_dicts.items() if v.index == 0}


class Bigram(BaseModel):
    el1: Lexeme
    el2: Lexeme

    class Config:
        allow_mutation = False
        frozen = True


@dataclass
class BigramData:
    bigrams_to_freqs: CounterType[Bigram] = Field(default_factory=Counter)
    bigrams_to_locations: Dict[Bigram, Set[Tuple[LineIndex, TokenIndex]]] = Field(
        default_factory=lambda: defaultdict(set)
    )
    left_lex_freqs: Dict[Lexeme, PositiveInt] = Field(default_factory=dict)
    right_lex_freqs: Dict[Lexeme, PositiveInt] = Field(default_factory=dict)

    @classmethod
    def from_lexemes(cls, lexeme_data: LexemeData) -> "BigramData":
        bigram_data = cls()
        corpus_iter_progress = tqdm(
            lexeme_data.line_lengths.items(),
            desc="Creating BigramData from LexemeData",
            unit="line",
            total=len(lexeme_data.line_lengths),
        )
        for line_ix, line_length in corpus_iter_progress:
            # Note: Here this means that we'll stop bigrams at the last term
            # because range(length - 1) means we wont add one index to the right.
            line_lexeme_data = lexeme_data.locations_to_root_lexemes(line_ix)
            line_items = list(line_lexeme_data.items())
            for (left_ix, left), (_, right) in zip(line_items, line_items[1:]):
                bigram = Bigram(el1=left, el2=right)
                location = (LineIndex(line_ix), TokenIndex(left_ix))
                bigram_data.add_bigram(bigram, location)
        return bigram_data

    def add_bigram(
        self, bigram: Bigram, location: Tuple[LineIndex, TokenIndex]
    ) -> None:
        # Original code has a conditional here to only do this if the bigram
        # isn't in the frequency counter yet, but that's a lot of checks that `add` can
        # do for us anyway
        if self.left_lex_freqs.get(bigram.el1, None):
            self.left_lex_freqs[bigram.el1] += 1
        else:
            self.left_lex_freqs[bigram.el1] = 1
        if self.right_lex_freqs.get(bigram.el2, None):
            self.right_lex_freqs[bigram.el2] += 1
        else:
            self.right_lex_freqs[bigram.el2] = 1
        self.bigrams_to_freqs[bigram] += 1
        self.bigrams_to_locations[bigram].add(location)

    @property
    def type_count(self) -> int:
        return len(self.bigrams_to_freqs)

    @property
    def bigram_count(self) -> int:
        return sum(self.bigrams_to_freqs.values())

    def remove_bigram(self, bigram: Bigram):
        freq = self.bigrams_to_freqs.pop(bigram)
        self.left_lex_freqs[bigram.el1] -= freq
        self.right_lex_freqs[bigram.el2] -= freq


lexemes = LexemeData.from_corpus(corpus)
bigrams = BigramData.from_lexemes(lexemes)


@dataclass
class WinnerInfo:
    bigram: Bigram
    merged_lexeme: Lexeme
    bigram_locations: List[Tuple[LineIndex, TokenIndex]]
    # merge_token_count: int = 0

    @classmethod
    def from_bigram_with_data(cls, bigram: Bigram, bigram_data: BigramData):
        el1_words = list(bigram.el1.word)
        el2_words = list(bigram.el2.word)
        all_words = el1_words + el2_words
        new_lexeme = Lexeme(word=tuple(all_words), index=0)
        locations = sorted(bigram_data.bigrams_to_locations[bigram])
        return cls(bigram=bigram, merged_lexeme=new_lexeme, bigram_locations=locations)  # type: ignore

    def clean_bigram_locations(self):
        """This is greedily selecting correct bigrams from the candidate locations of bigrams.

        Why? Well, in the case of a sentence like (a, a, a), with winner = (a, a), we can only convert
        the first occurrence of this bigram and not the second, since the first occurence would be transformed into the bigram,
        the new bigram in the second position no longer exists - but could be a candidate for the next round if it is indeed that common
        of a pattern.

        A more complex example is with winner (a, b, a, b) in ((a, b), (a, b), (a, b)). Here is the same idea: once we
        merge the first occurence it is no longer available, even though it occurs later.
        """
        clean_locations = []
        for line, location in groupby(self.bigram_locations, key=lambda x: x[0]):
            exclude_token = set()
            token_ix = [i[1] for i in location]
            for token in token_ix:
                if token in exclude_token:
                    continue
                excludes = [i for i in token_ix if i < token + self.n_lexemes]
                exclude_token.update(excludes)
                clean_locations.append((line, token))
            return clean_locations

    def satellites(self):
        return range(len(self.merged_lexeme.word))

    @property
    def n_lexemes(self) -> int:
        return len(self.merged_lexeme.word)

    @property
    def merge_token_count(self) -> int:
        return len(self.clean_bigram_locations())


winner: WinnerInfo = WinnerInfo.from_bigram_with_data(
    bigram=bigrams.bigrams_to_freqs.most_common()[0][0], bigram_data=bigrams
)  # fix


# Issue: We need to do this "all at once" or in some weird iterative order
# e.g. `a b a b` would have new bigrams `a b` and `a b`, but they would form the
# right and left context of each other.
# Correction for consecutive pairs (e.g. [(0, 1), (0, 3)])
# In `(0, 1)` - correct 'new_right' to have the new bigram as the right element (el2)
# In `(0, 3)`` - correct 'new_left' to have the new bigram as the left element

merged_tokens = []
correct_next = False
for (line_ix, word_ix), (next_line_ix, next_word_ix) in zip_longest(
    winner.clean_bigram_locations(),
    winner.clean_bigram_locations()[1:],
    fillvalue=(None, None),
):
    new_left = None
    new_right = None
    old_left = None
    old_right = None
    if line_ix != next_line_ix and isinstance(next_line_ix, int):
        # reset this every line for sanity
        # make sure it's not None
        # (on the last iteration, we still might need to correct left)
        correct_next = False
    left_context = lexemes.get(line_ix, word_ix - 1)
    if left_context:
        # There can be no left context if word_ix == 0.
        new_left = Bigram(el1=left_context, el2=winner.merged_lexeme)
        old_left = Bigram(el1=left_context, el2=winner.bigram.el1)
    right_context = lexemes.get(line_ix, word_ix + winner.n_lexemes)
    if right_context:
        # There can be no left context if word_ix + n_lexemes > line_length
        new_right = Bigram(el1=winner.merged_lexeme, el2=right_context)
        old_right = Bigram(
            el1=winner.bigram.el2,
            el2=right_context,
        )
    # We use a lookahead to correct consecutive occurences of the winner
    if correct_next:
        new_left = Bigram(el1=winner.merged_lexeme, el2=winner.merged_lexeme)
        correct_next = False

    if line_ix == next_line_ix and word_ix + winner.n_lexemes == next_word_ix:  # type: ignore
        correct_next = True
        new_right = Bigram(el1=winner.merged_lexeme, el2=winner.merged_lexeme)

    # # this is currently double counting...
    # if left_context is not None:
    #     pos = (LineIndex(line_ix), TokenIndex(word_ix - 1))
    #     conflicting_bigrams.add_bigram(old_left, pos)
    #     new_bigram = new_bigrams.add_bigram(new_left, pos)
    # if right_context is not None:
    #     pos = (LineIndex(line_ix), TokenIndex(word_ix))
    #     conflicting_bigrams.add_bigram(old_right, pos)
    #     new_bigrams.add_bigram(new_right, pos)

    # update lexeme locations
    # update_all_lexemes_with_merge_tokens
    for lexeme_index in range(winner.n_lexemes):
        if line_ix is None or word_ix is None:
            continue
        pos = TokenIndex(word_ix + lexeme_index)
        old_lexeme = lexemes.locations_to_lexemes[line_ix][pos]
        lexeme = Lexeme(word=winner.merged_lexeme.word, index=lexeme_index)
        lexemes.lexemes_to_locations[lexeme].add((LineIndex(line_ix), pos))
        lexemes.locations_to_lexemes[line_ix][pos] = lexeme
        lexemes.lexemes_to_locations[old_lexeme].remove((line_ix, pos))

# Add merged lexeme
lexemes.lexemes_to_freqs[winner.merged_lexeme] = winner.merge_token_count

# remove freq of individual elements
# update_lexeme_counts_with_merged_elements
el1_freq = lexemes.lexemes_to_freqs[winner.bigram.el1]
new_el1_freq = el1_freq - winner.merge_token_count
lexemes.lexemes_to_freqs[winner.bigram.el1] = new_el1_freq

el2_freq = lexemes.lexemes_to_freqs[winner.bigram.el2]
new_el2_freq = el2_freq - winner.merge_token_count
lexemes.lexemes_to_freqs[winner.bigram.el2] = new_el2_freq

# cleanup
lexemes.lexemes_to_freqs = {k: v for k, v in lexemes.lexemes_to_freqs.items() if v != 0}
lexemes.lexemes_to_locations = defaultdict(
    set, {k: v for k, v in lexemes.lexemes_to_locations.items() if v != set()}
)

bigrams2 = BigramData.from_lexemes(lexemes)
winner2: WinnerInfo = WinnerInfo.from_bigram_with_data(
    bigram=bigrams2.bigrams_to_freqs.most_common()[0][0], bigram_data=bigrams2
)  # fix

merged_tokens = []
correct_next = False
for (line_ix, word_ix), (next_line_ix, next_word_ix) in zip_longest(
    winner2.clean_bigram_locations(),
    winner2.clean_bigram_locations()[1:],
    fillvalue=(None, None),
):
    new_left = None
    new_right = None
    old_left = None
    old_right = None
    if line_ix != next_line_ix and isinstance(next_line_ix, int):
        # reset this every line for sanity
        # make sure it's not None
        # (on the last iteration, we still might need to correct left)
        correct_next = False
    left_context = lexemes.get(line_ix, word_ix - 1)
    if left_context:
        # There can be no left context if word_ix == 0.
        new_left = Bigram(el1=left_context, el2=winner2.merged_lexeme)
        old_left = Bigram(el1=left_context, el2=winner2.bigram.el1)
    right_context = lexemes.get(line_ix, word_ix + winner2.n_lexemes)
    if right_context:
        # There can be no left context if word_ix + n_lexemes > line_length
        new_right = Bigram(el1=winner2.merged_lexeme, el2=right_context)
        old_right = Bigram(
            el1=winner2.bigram.el2,
            el2=right_context,
        )
    # We use a lookahead to correct consecutive occurences of the winner2
    if correct_next:
        new_left = Bigram(el1=winner2.merged_lexeme, el2=winner2.merged_lexeme)
        correct_next = False

    if line_ix == next_line_ix and word_ix + winner2.n_lexemes == next_word_ix:  # type: ignore
        correct_next = True
        new_right = Bigram(el1=winner2.merged_lexeme, el2=winner2.merged_lexeme)

    # # this is currently double counting...
    # if left_context is not None:
    #     pos = (LineIndex(line_ix), TokenIndex(word_ix - 1))
    #     conflicting_bigrams.add_bigram(old_left, pos)
    #     new_bigram = new_bigrams.add_bigram(new_left, pos)
    # if right_context is not None:
    #     pos = (LineIndex(line_ix), TokenIndex(word_ix))
    #     conflicting_bigrams.add_bigram(old_right, pos)
    #     new_bigrams.add_bigram(new_right, pos)

    # update lexeme locations
    # update_all_lexemes_with_merge_tokens
    for lexeme_index in range(winner2.n_lexemes):
        if line_ix is None or word_ix is None:
            continue
        pos = TokenIndex(word_ix + lexeme_index)
        old_lexeme = lexemes.locations_to_lexemes[line_ix][pos]
        lexeme = Lexeme(word=winner2.merged_lexeme.word, index=lexeme_index)
        lexemes.lexemes_to_locations[lexeme].add((LineIndex(line_ix), pos))
        lexemes.locations_to_lexemes[line_ix][pos] = lexeme
        lexemes.lexemes_to_locations[old_lexeme].remove((line_ix, pos))

# Add merged lexeme
lexemes.lexemes_to_freqs[winner2.merged_lexeme] = winner2.merge_token_count

# remove freq of individual elements
# update_lexeme_counts_with_merged_elements
el1_freq = lexemes.lexemes_to_freqs[winner2.bigram.el1]
new_el1_freq = el1_freq - winner2.merge_token_count
lexemes.lexemes_to_freqs[winner2.bigram.el1] = new_el1_freq

el2_freq = lexemes.lexemes_to_freqs[winner2.bigram.el2]
new_el2_freq = el2_freq - winner2.merge_token_count
lexemes.lexemes_to_freqs[winner2.bigram.el2] = new_el2_freq

# cleanup
lexemes.lexemes_to_freqs = {k: v for k, v in lexemes.lexemes_to_freqs.items() if v != 0}
lexemes.lexemes_to_locations = defaultdict(
    set, {k: v for k, v in lexemes.lexemes_to_locations.items() if v != set()}
)

bigrams3 = BigramData.from_lexemes(lexemes)
winner3: WinnerInfo = WinnerInfo.from_bigram_with_data(
    bigram=bigrams3.bigrams_to_freqs.most_common()[0][0], bigram_data=bigrams3
)  # fix
