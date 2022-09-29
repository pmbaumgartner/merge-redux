
Let's say we only have one character and we're going to run 3 iterations on a corpus with a single text `a a a a`. This is what it would look like after 2 iterations.


```
((a,), 0), ((a,), 0), ((a,), 0), ((a,), 0)
((a,a), 0), ((a,a), 1), ((a,a), 0), ((a,a), 1)
((a,a,a,a), 0), ((a,a,a,a), 1), ((a,a,a,a), 2), ((a,a,a,a), 3)
```

# Note: Failure case when there's only one lexeme term left? Figure this out.

# Collect bigrams and statistics

```python
i = 1
# Location: LineIndex, TokenIndex
lexemes_to_locations = {(("a",), 0): {(0, 0), (0, 1), (0, 2), (0, 3)}}

# Calculate all bigrams:
bigram_counts = {Bigram(el1=(('a',), 0), el2=(('a',), 0)): 3}
bigram_locations = {(0, 1), (0, 2), (0, 0)}
```
We also need to track the left and right elements individually, so that our numbers add up correctly to calculate the winner.
# TODO: We also track which bigrams each element maps to, but isn't that self-evident from the bigram? 

Then we calculate the winner. Here the winner is (obviously) (('a',), 0). So now what happens? 
We find the locations of the winning bigrams.

Make a new Lexeme: the merged bigram words, plus the position.

(a,), 0), ((a,), 0) -> ((a,a), 0), ((a,a), 1)
((a,a), 0), ((a,a), 1), ((a,a), 0), ((a,a), 1) -> ((a,a,a,a), 0), ((a,a,a,a), 1), ((a,a,a,a), 2), ((a,a,a,a), 3)


# bigram step

sent = (a, b, c, d)
winner = (b, c)
new = (a, (b,c)), ((b,c), d)
conflict_old = (a, b), (c, d)

sent = (a, b, a, b)
winner = (a, b)
new = ((a, b), (a, b))
conflict_old = (b, a) x 2 [TODO: CHECK]



Do we need location updates? I don't think so because if you know the length of a lexeme (i.e. words), and you know where it starts, you can always jump to pos + length to get the second token in the lexeme. When would you need to know the length? 


((a, b), 0), ((a, b), 1), ((c, d), 0), ((c, d), 1)

((a, b, c, d), 0), ((a, b, c, d), 1), ((a, b, c, d), 2), ((a, b, c, d), 3) 



An open question I have with this - why am I having such a hard time implementing such a straightforward algorithm?
Theories:
- This is the wrong data structure(s) to attempt to do this problem
- I'm anchored in the existing code in how I attempt to solve the problem
- The recursive/merging nature of the problem
- There are two related data structures that are difficult to keep in sync


This last part is key, I figured out I can just update one thing (the lexemes) and then recreate the bigrams from that lexeme data - since I already have that function. Then I just have to make sure the lexeme data is correct and can be parsed correctly into the bigram structure.


# Issue: We need to do this "all at once" or in some weird iterative order
# e.g. `a b a b` would have new bigrams `a b` and `a b`, but they would form the
# right and left context of each other.
# Correction for consecutive pairs (e.g. [(0, 1), (0, 3)])
# In `(0, 1)` - correct 'new_right' to have the new bigram as the right element (el2)
# In `(0, 3)`` - correct 'new_left' to have the new bigram as the left element