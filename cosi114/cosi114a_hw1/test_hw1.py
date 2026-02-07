#! /usr/bin/env python
# Version 1.0.1

import unittest
from collections import Counter
from pathlib import Path
from typing import Generator, Iterable, Iterator, Sequence, TypeVar, Union

from grader import Grader, points
from hw1 import (
    END_TOKEN,
    START_TOKEN,
    bigram_counts,
    bigram_probs,
    bigrams,
    counts_to_probs,
    trigram_counts,
    trigram_probs,
    trigrams,
    unigram_counts,
    unigram_probs,
)

T = TypeVar("T")

TEST_DATA = Path("test_data")
SAMPLE_LYRICS = TEST_DATA / "costello_radio.txt"


def sample_sentence() -> list[str]:
    """Return a sample sentence."""
    return [
        "The",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
        ".",
    ]


def load_tokenized_file(path: Union[Path, str]) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            assert tokens, "Empty line in input"
            yield tuple(tokens)


class DefensiveIterable(Iterable[T]):
    def __init__(self, source: Iterable[T]):
        self.source: Iterable[T] = source

    def __iter__(self) -> Iterator[T]:
        return iter(self.source)

    def __len__(self) -> int:
        # This object should never be put into a sequence, so we sabotage the
        # __len__ function to make it difficult to do so. We specifically raise
        # ValueError because TypeError and NotImplementedError appear to be
        # handled by the list function.
        raise ValueError(
            "You cannot put this iterable into a sequence (list, tuple, etc.). "
            "Instead, iterate over it using a for loop."
        )


def load_tokenized_test_file(path: Union[str, Path]) -> Iterable[Sequence[str]]:
    """Return a defensive iterable over sentences as sequences of tokens."""
    return DefensiveIterable(load_tokenized_file(path))


class TestNGrams(unittest.TestCase):
    @points(1)
    def test_type_bigrams(self) -> None:
        """Bigrams should return a list of length-two tuples."""
        ngrams = bigrams(sample_sentence())
        self.assertEqual(list, type(ngrams))
        self.assertEqual(tuple, type(ngrams[0]))
        self.assertEqual(2, len(ngrams[0]))

    @points(5)
    def test_bigrams(self) -> None:
        """Return correct bigrams for test sentence 1."""
        self.assertEqual(
            [
                (START_TOKEN, "The"),
                ("The", "quick"),
                ("quick", "brown"),
                ("brown", "fox"),
                ("fox", "jumps"),
                ("jumps", "over"),
                ("over", "the"),
                ("the", "lazy"),
                ("lazy", "dog"),
                ("dog", "."),
                (".", END_TOKEN),
            ],
            bigrams(sample_sentence()),
        )

    @points(1)
    def test_type_trigrams(self) -> None:
        """Trigrams should return a list of length-three tuples."""
        ngrams = trigrams(sample_sentence())
        self.assertEqual(list, type(ngrams))
        self.assertEqual(tuple, type(ngrams[0]))
        self.assertEqual(3, len(ngrams[0]))

    @points(5)
    def test_trigrams(self) -> None:
        """Trigrams are correct for test sentence 1."""
        self.assertEqual(
            [
                (START_TOKEN, START_TOKEN, "The"),
                (START_TOKEN, "The", "quick"),
                ("The", "quick", "brown"),
                ("quick", "brown", "fox"),
                ("brown", "fox", "jumps"),
                ("fox", "jumps", "over"),
                ("jumps", "over", "the"),
                ("over", "the", "lazy"),
                ("the", "lazy", "dog"),
                ("lazy", "dog", "."),
                ("dog", ".", END_TOKEN),
            ],
            trigrams(sample_sentence()),
        )


class TestCounts(unittest.TestCase):
    AA_BB_CC_SENTENCES = [["aa"], ["aa", "bb"], ["aa", "bb"], ["aa", "cc"]]

    @points(2)
    def test_unigram_counts(self) -> None:
        """Simple unigram counts are correct."""
        counts = unigram_counts(self.AA_BB_CC_SENTENCES)
        # Check for exact type dict, not defaultdict
        self.assertEqual(dict, type(counts))
        self.assertEqual({"aa": 4, "bb": 2, "cc": 1}, counts)

    @points(3)
    def test_bigram_counts(self) -> None:
        """Simple bigram counts are correct."""
        counts = bigram_counts(self.AA_BB_CC_SENTENCES)
        # Check for exact type dict, not defaultdict
        self.assertEqual(dict, type(counts))
        for value in counts.values():
            # Check for exact type dict, not defaultdict
            self.assertEqual(dict, type(value))
        self.assertEqual(
            {
                START_TOKEN: {"aa": 4},
                "aa": {"bb": 2, "cc": 1, END_TOKEN: 1},
                "bb": {END_TOKEN: 2},
                "cc": {END_TOKEN: 1},
            },
            counts,
        )

    @points(3)
    def test_trigram_counts(self) -> None:
        """Simple trigram counts are correct."""
        counts = trigram_counts(self.AA_BB_CC_SENTENCES)
        # Check for exact type dict, not defaultdict
        self.assertEqual(dict, type(counts))
        for value in counts.values():
            # Check for exact type dict, not defaultdict
            self.assertEqual(dict, type(value))
        self.assertEqual(
            {
                (START_TOKEN, START_TOKEN): {"aa": 4},
                (START_TOKEN, "aa"): {"bb": 2, "cc": 1, END_TOKEN: 1},
                ("aa", "bb"): {END_TOKEN: 2},
                ("aa", "cc"): {END_TOKEN: 1},
            },
            counts,
        )


class TestCountsToProbs(unittest.TestCase):
    @points(2)
    def test_counts_to_probs_type(self) -> None:
        """Probability dictionary is a dict[T, float]."""
        counts = Counter(["cat", "dog", "dog"])
        probs = counts_to_probs(counts)
        # We do a strict type check instead of isinstance because we don't want defaultdict
        self.assertEqual(dict, type(probs))
        # Get a single key/value pair
        key, val = next(iter(probs.items()))
        self.assertEqual(str, type(key))
        self.assertEqual(float, type(val))

    @points(7)
    def test_counts_to_probs_values(self) -> None:
        """Basic probabilities are correct."""
        counts = Counter(["cat", "dog", "dog"])
        probs = counts_to_probs(counts)
        self.assertAlmostEqual(0.3333, probs["cat"], places=3)
        self.assertAlmostEqual(0.6666, probs["dog"], places=3)


class TestUnigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_test_file(SAMPLE_LYRICS)

    @points(1)
    def test_unigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the correct number of keys."""
        counts = unigram_counts(self.sentences_gen)
        probs = unigram_probs(counts)
        self.assertEqual(35, len(probs))

    @points(1)
    def test_unigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        counts = unigram_counts(self.sentences_gen)
        probs = unigram_probs(counts)

        self.assertAlmostEqual(19 / 73, probs["radio"])
        self.assertAlmostEqual(3 / 73, probs["they"])
        self.assertAlmostEqual(1 / 73, probs["treason"])

    @points(1)
    def test_unigram_probs_type(self) -> None:
        """Keys and values have the correct types."""
        counts = unigram_counts(self.sentences_gen)
        probs = unigram_probs(counts)

        # Check key and value types
        self.assertEqual(dict, type(probs))
        key, val = next(iter(probs.items()))
        self.assertEqual(str, type(key))
        self.assertEqual(float, type(val))


class TestBigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_test_file(SAMPLE_LYRICS)

    @points(2)
    def test_bigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the correct number of keys."""
        counts = bigram_counts(self.sentences_gen)
        probs = bigram_probs(counts)
        self.assertEqual(36, len(probs))

    @points(5)
    def test_bigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        counts = bigram_counts(self.sentences_gen)
        probs = bigram_probs(counts)
        you = probs["you"]
        self.assertEqual(4, len(you))
        self.assertAlmostEqual(0.4, you["better"])
        self.assertAlmostEqual(0.2, you["any"])
        self.assertAlmostEqual(0.2, you["had"])
        self.assertAlmostEqual(0.2, you["are"])
        radio = probs["radio"]
        self.assertAlmostEqual(9 / 19, radio[END_TOKEN])

    @points(2)
    def test_bigram_probs_type(self) -> None:
        """Inner and outer dicts have the correct types."""
        counts = bigram_counts(self.sentences_gen)
        probs = bigram_probs(counts)

        # Check outer dict type
        self.assertEqual(dict, type(probs))
        outer_key, inner_dict = next(iter(probs.items()))
        self.assertEqual(str, type(outer_key))

        # Check inner dict
        self.assertEqual(dict, type(inner_dict))
        inner_key, inner_val = next(iter(inner_dict.items()))
        self.assertEqual(str, type(inner_key))
        self.assertEqual(float, type(inner_val))


class TestTrigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_test_file(SAMPLE_LYRICS)

    @points(2)
    def test_trigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the correct number of keys."""
        counts = trigram_counts(self.sentences_gen)
        probs = trigram_probs(counts)
        self.assertEqual(46, len(probs))

    @points(5)
    def test_trigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        counts = trigram_counts(self.sentences_gen)
        probs = trigram_probs(counts)
        radio = probs[(START_TOKEN, "radio")]
        self.assertEqual(2, len(radio))
        self.assertAlmostEqual(0.8, radio[","])
        self.assertAlmostEqual(0.2, radio["is"])
        radio = probs[(",", "radio")]
        self.assertAlmostEqual(1.0, radio[END_TOKEN])

    @points(2)
    def test_trigram_probs_type(self) -> None:
        """Inner and outer dicts have the correct types."""
        counts = trigram_counts(self.sentences_gen)
        probs = trigram_probs(counts)

        # Check outer dict
        self.assertEqual(dict, type(probs))
        outer_key, inner_dict = next(iter(probs.items()))
        self.assertEqual(tuple, type(outer_key))
        self.assertEqual(2, len(outer_key))
        self.assertEqual(str, type(outer_key[0]))
        self.assertEqual(str, type(outer_key[1]))

        # Check inner dict
        self.assertEqual(dict, type(inner_dict))
        inner_key, inner_val = next(iter(inner_dict.items()))
        self.assertEqual(str, type(inner_key))
        self.assertEqual(float, type(inner_val))


def main() -> None:
    tests = [
        TestCountsToProbs,
        TestNGrams,
        TestCounts,
        TestUnigramProbs,
        TestBigramProbs,
        TestTrigramProbs,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
