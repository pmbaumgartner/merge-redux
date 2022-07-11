- investigate [bounter](https://github.com/RaRe-Technologies/bounter) for counting
- consistent naming (`bgr`/`bigram`)
- new steps should copy and modify, not overwrite original data (for debugging)
- replace `del` with `dict.pop(key, None)`, so that we can conditionally check that a value has been removed
- performance
  1. update_conflicting_bigram_freqs (update_cell_value)
  2. calculate (calc_ll_for_single_table)
  3. main_control_loop (create_bigrams_with_lexemes_surrounding_satellite)
- multiprocessing?
- removing single occurence Lexemes and Bigrams from BigramData.
  - Replace with a single `PRUNED` lexeme 
- `WinningBigram` data structure (merge token count, merged_token: Lexeme, locations, etc)