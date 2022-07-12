# flake8: noqa
from src.merge_redux import __version__
from src.merge_redux import run
from src.merge_redux.core import Lexeme, Word

from .fixtures import sample_corpus


def test_version():
    assert __version__ == "0.1.0"


def test_single_iter(sample_corpus):
    winners = run(sample_corpus, 0, 1)
    assert winners == [Lexeme((Word("you", 0), Word("know", 1)), 0)]
