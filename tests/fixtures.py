from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def sample_corpus():
    """We'll use the corpus included in the original implementation,
    which is "[...] a combination of the Santa Barbara
    Corpus of Spoken American English and the spoken
    component of the ICE Canada corpus"

    > Du Bois, John W., Wallace L. Chafe, Charles Meyers, Sandra A. Thompson, Nii Martey, and Robert Englebretson (2005). Santa Barbara corpus of spoken American English. Philadelphia: Linguistic Data Consortium.

    > Newman, John and Georgie Columbus (2010). The International Corpus of English – Canada. Edmonton, Alberta: University of Alberta.

    Note that these are dialogue corpuses, so each line is often referred to as a `turn`. They've also been pre-processed with lowercasing, and punctuation replaced with alphanumeric substitutions (`_` --> `undrscr`).
    """
    corpus: List[List[str]] = []
    this_folder = Path(__file__).parent
    for txt_file in (this_folder / Path("sample_corpus/")).glob("*.TXT"):
        for line in txt_file.read_text().split("\n"):
            if line:
                tokens: List[str] = line.split(" ")
                corpus.append(tokens)
    return corpus
