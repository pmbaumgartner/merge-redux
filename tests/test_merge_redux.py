# flake8: noqa
from src.merge_redux import __version__
from src.merge_redux import run
from src.merge_redux.core import Lexeme

from .fixtures import sample_corpus


def test_version():
    assert __version__ == "0.1.0"


def test_single_iter(sample_corpus):
    winners = run(sample_corpus, 1)
    assert winners[0].merged_lexeme == Lexeme(("you", "know"), 0)


# a a a a

# c a b a b a b d
