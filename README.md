# MERGE Multi-Word Expression Algorithm: Re-Implementation

⚠️ WIP

This is a re-implementation of a multi-word expression (MWE) discovery algorithm called MERGE, detailed in a publication and PhD thesis[^2][^3]. The code was derived from an existing implementation from the original author[^1], and was coverted to a more modern python style. This included converting from python 2 to 3, the inclusion of types, and modifications to make the code a bit more linear rather than the original object-oriented implementation.

In its current state is it not optimized - for the most part it replicates the data structures and logic of the original implementation and serves mostly as a translation of that code.

**Usage**

```python
from typing import List
from merge_redux import run as merge_run

corpus: List[List[str]] = [
    ["a", "list", "of", "already", "tokenized", "texts"],
    ["where", "each", "item", "is", "a", "list", "of", "tokens"],
]

winners = merge_run(corpus, gapsize=1, iterations=1)

# winners == [Lexeme((Word('a', 0), Word('of', 2)), 0)]
# This means the winner of iteration 1 was a Lexeme where the
# first word is 'a' in position 0 and a second token with
# the word 'of' in position 2. We have succesfully captured a
# MWE with a discontinuity of the form "a _ of"
```

Wanna go faster? You can pass `min_bigram_freq` and `min_lexeme_freq` to `run`, which will ignore any bigrams or lexemes in any iteration with a total frequency in the corpus lower than those values, which can result in a significant speedup when calculating the log-likelihood statistics for each element in the corpus. For example, the example below was about a 10x speedup on the sample corpus:

```python
winners = run(corpus, 1, 100, min_bigram_freq=10, min_lexeme_freq=10)
```

**Install**:

```
pip install git+https://github.com/pmbaumgartner/merge-redux.git 
```

**Limitations**

**No tie-breaking** - I found this while testing and comparing to the original reference. If two bigrams are tied for log-likelihood, there is no tie-breaking mechanism. Both this implementation and the original implementation simply pick the first bigram from the index with the maximum log-likelihood value. However, we have slightly different implementations of how the statistics table is created (i.e., the ordering of the index), which makes direct comparisons between implementations difficult.

**Single Bigrams with discontinuities forming from distinct Lexeme positions** - This example I am less sure that I have implemented the algorithm correctly, but I think there is an issue with what can constitute a "bigram" with discontinuities. 

For an example corpus with a single document that looks like this:

```
# 3 0 2 1 3 0 3 1 4
[["3", "0", "2", "1", "3", "0", "3", "1", "4"]]
```

After two iterations, this implementation picks the following winner from this single document corpus: `["3" (0), "0" (1), "1" (3)]` (in format `("token", position)`).

On the second round a single bigram of `([("0", 0), ("1", 2)], [("3", 0)])`  gets created. If we think about it's creation, it is actually two distinct bigrams:

1. One from indices `(1, 3)`, where the second element occurs at position `4` (after the second lexeme)
2. One from indices `(5, 7)`, where the second element occurs at position `6` (after the first word, in the gap)

A byproduct of how the algorithm handles discontinuities means these both count towards the same source Bigram `([("0", 0), ("1", 2)], [("3", 0)])`. 

I think the algorithm is counting on the fact that there probably isn't an _actual_ sequence of lexemes where a word would appear withinin the gap **and** after the gap, and this is more of an artifact of this specific example with a single document and unrealistically small vocabulary.

**References**:

[^1]: awahl1, MERGE. 2017. Accessed: Jul. 11, 2022. [Online]. Available: https://github.com/awahl1/MERGE

[^2]: A. Wahl and S. Th. Gries, “Multi-word Expressions: A Novel Computational Approach to Their Bottom-Up Statistical Extraction,” in Lexical Collocation Analysis, P. Cantos-Gómez and M. Almela-Sánchez, Eds. Cham: Springer International Publishing, 2018, pp. 85–109. doi: 10.1007/978-3-319-92582-0_5.

[^3]: A. Wahl, “The Distributional Learning of Multi-Word Expressions: A Computational Approach,” p. 190.
