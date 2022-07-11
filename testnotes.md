
# After 2 iters, winner: ["3" (0), "0" (1), "1" (3)]
```python

corpus: List[List[str]] = [["3", "0", "2", "1", "3", "0", "3", "1", "4"]]
```

# Wrong
# hints:
- Iter 1 gets calculated correctly
- Winner gets calculated correctly at the beginning of iter 2
- WINNER IS GETTING CALCULATED W/ REVERSED ELEMENTS?
   - Maybe not, it's just picking another common bigram
```
Bigram(el1=Lexeme((Word('0', 0), Word('1', 2)), 0), el2=Lexeme((Word('3', 0),), 0), gapsize=0)
```

el1 and 2 should be flipped



- Missing new locations in new_bigrams.bigrams_to_locations?
  - No nevermind


It's actually because they're tied for log-likelihood and it's going to pick the one at the top. In their implementation, the one at the top.


Undesirable behavior:
with `corpus = [["3", "0", "2", "1", "3", "0", "3", "1", "4"]]`, on the second round a single bigram of `(((Word('0', 0), Word('1', 2)), 0), ((Word('3', 0),), 0), 2)` gets created. If we think about it's creation, it is actually two distinct bigrams:

1. On indices (1, 3), where it occurs at position 4 (after the second word)
2. On indices (5, 7), where it occurs at position 6 (after the first word, in the gap)

A byproduct of how the algorithm handles discontinuities means these both count towards the same source Bigram `[("0", 0), ("1", 2)], [("3", 0)]`. I think the algorithm is counting on the fact that there probably isn't an _actual_ sequence of lexemes where a word would appear withinin the gap **and** after the gap.
