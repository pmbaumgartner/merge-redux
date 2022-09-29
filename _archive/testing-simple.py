from merge_redux import run

corpus = [["a", "a", "a", "a"]]
# Wrong

winners = run(corpus, 0, 2, min_bigram_freq=0, min_lexeme_freq=0)
