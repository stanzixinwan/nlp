#! /usr/bin/env python

import random
import unittest
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar, Union

from grader import Grader, points
from hw2 import (
    END_TOKEN,
    START_TOKEN,
    BigramInterpolation,
    BigramMLE,
    ProbabilityDistribution,
    TrigramMLE,
    UnigramLidstoneSmoothed,
    UnigramMLE,
    bigram_perplexity,
    bigram_sequence_prob,
    load_tokenized_file,
    sample_bigrams,
    sample_trigrams,
    trigram_perplexity,
    trigram_sequence_prob,
    unigram_perplexity,
    unigram_sequence_prob,
)

T = TypeVar("T")
TEST_DATA = Path("test_data")
SAMPLE_LYRICS = TEST_DATA / "costello_radio.txt"
NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]


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


class DictProbabilityDistribution(ProbabilityDistribution[T]):
    def __init__(self, probs: dict[T, float]):
        self._probs: dict[T, float] = probs

    def prob(self, item: T) -> float:
        return self._probs.get(item, 0.0)


class SeedControlledTestCase(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(4)


class TestSampleBigrams(SeedControlledTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Probs for test_data/costello_radio.txt
        self.probs = {
            START_TOKEN: {
                "radio": 0.7142857142857143,
                "they": 0.07142857142857142,
                "but": 0.07142857142857142,
                "so": 0.07142857142857142,
                "you": 0.07142857142857142,
            },
            "radio": {
                "is": 0.10526315789473684,
                END_TOKEN: 0.47368421052631576,
                ",": 0.42105263157894735,
            },
            "is": {"a": 0.5, "cleaning": 0.5},
            "a": {"sound": 1.0},
            "sound": {"salvation": 1.0},
            "salvation": {END_TOKEN: 1.0},
            "cleaning": {"up": 1.0},
            "up": {"the": 1.0},
            "the": {
                "nation": 0.3333333333333333,
                "voice": 0.3333333333333333,
                "radio": 0.3333333333333333,
            },
            "nation": {END_TOKEN: 1.0},
            "they": {
                "say": 0.3333333333333333,
                "don't": 0.3333333333333333,
                "think": 0.3333333333333333,
            },
            "say": {"you": 1.0},
            "you": {"better": 0.4, "any": 0.2, "had": 0.2, "are": 0.2},
            "better": {"listen": 0.6666666666666666, "do": 0.3333333333333333},
            "listen": {"to": 1.0},
            "to": {"the": 1.0},
            "voice": {"of": 1.0},
            "of": {"reason": 1.0},
            "reason": {END_TOKEN: 1.0},
            "but": {"they": 1.0},
            "don't": {"give": 1.0},
            "give": {"you": 1.0},
            "any": {"choice": 1.0},
            "choice": {"'cause": 1.0},
            "'cause": {"they": 1.0},
            "think": {"that": 1.0},
            "that": {"it's": 1.0},
            "it's": {"treason": 1.0},
            "treason": {END_TOKEN: 1.0},
            "so": {"you": 1.0},
            "had": {"better": 1.0},
            "do": {"as": 1.0},
            "as": {"you": 1.0},
            "are": {"told": 1.0},
            "told": {END_TOKEN: 1.0},
            ",": {"radio": 1.0},
        }

    @points(5)
    def test_first_sample(self) -> None:
        """The first bigram-generated sentence is correct."""
        sent = sample_bigrams(self.probs)
        self.assertEqual(["radio", ",", "radio", ",", "radio", ",", "radio"], sent)

    @points(4)
    def test_second_sample(self) -> None:
        """The second bigram-generated sentence is correct."""
        # Throw away one sample
        sample_bigrams(self.probs)
        sent = sample_bigrams(self.probs)
        self.assertEqual(
            [
                "radio",
                ",",
                "radio",
                ",",
                "radio",
                ",",
                "radio",
                "is",
                "cleaning",
                "up",
                "the",
                "nation",
            ],
            sent,
        )


class TestSampleTrigrams(SeedControlledTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Probs for test_data/costello_radio.txt
        self.probs = {
            (START_TOKEN, START_TOKEN): {
                "radio": 0.7142857142857143,
                "they": 0.07142857142857142,
                "but": 0.07142857142857142,
                "so": 0.07142857142857142,
                "you": 0.07142857142857142,
            },
            (START_TOKEN, "radio"): {"is": 0.2, ",": 0.8},
            ("radio", "is"): {"a": 0.5, "cleaning": 0.5},
            ("is", "a"): {"sound": 1.0},
            ("a", "sound"): {"salvation": 1.0},
            ("sound", "salvation"): {END_TOKEN: 1.0},
            ("is", "cleaning"): {"up": 1.0},
            ("cleaning", "up"): {"the": 1.0},
            ("up", "the"): {"nation": 1.0},
            ("the", "nation"): {END_TOKEN: 1.0},
            (START_TOKEN, "they"): {"say": 1.0},
            ("they", "say"): {"you": 1.0},
            ("say", "you"): {"better": 1.0},
            ("you", "better"): {"listen": 1.0},
            ("better", "listen"): {"to": 1.0},
            ("listen", "to"): {"the": 1.0},
            ("to", "the"): {"voice": 0.5, "radio": 0.5},
            ("the", "voice"): {"of": 1.0},
            ("voice", "of"): {"reason": 1.0},
            ("of", "reason"): {END_TOKEN: 1.0},
            (START_TOKEN, "but"): {"they": 1.0},
            ("but", "they"): {"don't": 1.0},
            ("they", "don't"): {"give": 1.0},
            ("don't", "give"): {"you": 1.0},
            ("give", "you"): {"any": 1.0},
            ("you", "any"): {"choice": 1.0},
            ("any", "choice"): {"'cause": 1.0},
            ("choice", "'cause"): {"they": 1.0},
            ("'cause", "they"): {"think": 1.0},
            ("they", "think"): {"that": 1.0},
            ("think", "that"): {"it's": 1.0},
            ("that", "it's"): {"treason": 1.0},
            ("it's", "treason"): {END_TOKEN: 1.0},
            (START_TOKEN, "so"): {"you": 1.0},
            ("so", "you"): {"had": 1.0},
            ("you", "had"): {"better": 1.0},
            ("had", "better"): {"do": 1.0},
            ("better", "do"): {"as": 1.0},
            ("do", "as"): {"you": 1.0},
            ("as", "you"): {"are": 1.0},
            ("you", "are"): {"told": 1.0},
            ("are", "told"): {END_TOKEN: 1.0},
            (START_TOKEN, "you"): {"better": 1.0},
            ("the", "radio"): {END_TOKEN: 1.0},
            ("radio", ","): {"radio": 1.0},
            (",", "radio"): {END_TOKEN: 1.0},
        }

    @points(5)
    def test_first_sample(self) -> None:
        """The first trigram-generated sentence is correct."""
        sent = sample_trigrams(self.probs)
        self.assertEqual(["radio", ",", "radio"], sent)

    @points(4)
    def test_third_sample(self) -> None:
        """The third trigram-generated sentence is correct."""
        # Throw away first two samples
        sample_trigrams(self.probs)
        sample_trigrams(self.probs)
        sent = sample_trigrams(self.probs)
        self.assertEqual(
            ["so", "you", "had", "better", "do", "as", "you", "are", "told"], sent
        )


class TestMLE(unittest.TestCase):
    @points(2)
    def test_unigram_simple(self) -> None:
        """Simple unigram probabilities are correct."""
        counts = {number: 1 for number in NUMBERS}
        mle = UnigramMLE(counts)
        self.assertAlmostEqual(1 / 10, mle.prob("two"))

    @points(2)
    def test_bigram_simple1(self) -> None:
        """Simple bigram probabilities are correct."""
        # Bigram counts for NUMBERS
        counts = {
            START_TOKEN: {"one": 1},
            "one": {"two": 1},
            "two": {"three": 1},
            "three": {"four": 1},
            "four": {"five": 1},
            "five": {"six": 1},
            "six": {"seven": 1},
            "seven": {"eight": 1},
            "eight": {"nine": 1},
            "nine": {"ten": 1},
            "ten": {END_TOKEN: 1},
        }
        mle = BigramMLE(counts)
        self.assertEqual(1.0, mle.prob(("two", "three")))

    @points(2)
    def test_bigram_simple2(self) -> None:
        """Simple bigram probabilities are correct."""
        # Counts and probs for:
        # [["aa", "bb"], ["aa", "bb"], ["aa", "cc"]]
        counts = {
            START_TOKEN: {"aa": 3},
            "aa": {"bb": 2, "cc": 1},
            "bb": {END_TOKEN: 2},
            "cc": {END_TOKEN: 1},
        }
        mle = BigramMLE(counts)
        self.assertAlmostEqual(2 / 3, mle.prob(("aa", "bb")))

    @points(2)
    def test_trigram_simple1(self) -> None:
        """Simple trigram probabilities are correct."""
        # Trigram counts for NUMBERS
        counts = {
            (START_TOKEN, START_TOKEN): {"one": 1},
            (START_TOKEN, "one"): {"two": 1},
            ("one", "two"): {"three": 1},
            ("two", "three"): {"four": 1},
            ("three", "four"): {"five": 1},
            ("four", "five"): {"six": 1},
            ("five", "six"): {"seven": 1},
            ("six", "seven"): {"eight": 1},
            ("seven", "eight"): {"nine": 1},
            ("eight", "nine"): {"ten": 1},
            ("nine", "ten"): {END_TOKEN: 1},
        }
        mle = TrigramMLE(counts)
        self.assertEqual(1.0, mle.prob(("two", "three", "four")))

    @points(2)
    def test_trigram_simple2(self) -> None:
        """Simple trigram probabilities are correct."""
        # Counts and probs for:
        # [["aa"], ["aa", "bb"], ["aa", "bb"], ["aa", "cc"]]
        counts = {
            (START_TOKEN, START_TOKEN): {"aa": 4},
            (START_TOKEN, "aa"): {"bb": 2, "cc": 1, END_TOKEN: 1},
            ("aa", "bb"): {END_TOKEN: 2},
            ("aa", "cc"): {END_TOKEN: 1},
        }
        mle = TrigramMLE(counts)
        self.assertAlmostEqual(2 / 4, mle.prob((START_TOKEN, "aa", "bb")))


class TestSequenceProb(unittest.TestCase):
    @points(3)
    def test_unigram_seq(self) -> None:
        """Simple unigram sequence probability is correct."""
        # Probs for:
        # [["aa", "bb"], ["aa", "bb"], ["aa", "cc"]]
        probs = {"aa": 3 / 6, "bb": 2 / 6, "cc": 1 / 6}
        dist = DictProbabilityDistribution(probs)
        self.assertAlmostEqual(-0.6931471805599453, unigram_sequence_prob(["aa"], dist))

    @points(3)
    def test_bigram_seq(self) -> None:
        """Simple bigram sequence probability is correct."""
        # Probs for:
        # [["aa", "bb"], ["aa", "bb", "aa"], ["aa", "cc"]]
        probs = {
            (START_TOKEN, "aa"): 1.0,
            ("aa", "bb"): 2 / 4,
            ("aa", "cc"): 1 / 4,
            ("aa", END_TOKEN): 1 / 4,
            ("bb", "aa"): 1 / 2,
            ("bb", END_TOKEN): 1 / 2,
            ("cc", END_TOKEN): 1.0,
        }
        dist = DictProbabilityDistribution(probs)
        self.assertAlmostEqual(
            -2.772588722239781, bigram_sequence_prob(["aa", "bb", "aa"], dist)
        )

    @points(3)
    def test_trigram_seq(self) -> None:
        """Simple trigram sequence probability is correct."""
        # Probs for:
        # [["aa", "bb"], ["aa", "bb", "aa", "bb"], ["aa", "cc"]]
        probs = {
            (START_TOKEN, START_TOKEN, "aa"): 1.0,
            (START_TOKEN, "aa", "bb"): 2 / 3,
            (START_TOKEN, "aa", "cc"): 1 / 3,
            ("aa", "bb", END_TOKEN): 2 / 3,
            ("aa", "bb", "aa"): 1 / 3,
            ("bb", "aa", "bb"): 1.0,
            ("aa", "cc", END_TOKEN): 1.0,
        }

        dist = DictProbabilityDistribution(probs)
        self.assertAlmostEqual(
            -1.9095425048844386, trigram_sequence_prob(["aa", "bb", "aa", "bb"], dist)
        )


class TestLidstoneSmoothing(unittest.TestCase):
    @points(3)
    def test_lidstone_simple(self) -> None:
        """Simple Lidstone probability is correct."""
        counts = {"zero": 0, "one": 1, "two": 2, "three": 3}
        dist = UnigramLidstoneSmoothed(counts, 1.0)
        self.assertAlmostEqual(2 / 10, dist.prob("one"))


class TestInterpolation(unittest.TestCase):
    # These were used to compute the unigram and bigram counts
    SEQUENCES = [
        ["aa"],
        ["aa", "bb"],
        ["aa", "cc"],
        ["dd"],
    ]

    @points(4)
    def test_interpolation_simple(self) -> None:
        """Simple interpolation between MLE probabilities is correct."""
        uni_counts = {"aa": 3, "bb": 1, "cc": 1, "dd": 1}
        uni_mle = UnigramMLE(uni_counts)
        bi_counts = {
            "<START>": {"aa": 3, "dd": 1},
            "aa": {"<END>": 1, "bb": 1, "cc": 1},
            "bb": {"<END>": 1},
            "cc": {"<END>": 1},
            "dd": {"<END>": 1},
        }
        bi_mle = BigramMLE(bi_counts)

        dist = BigramInterpolation(uni_mle, bi_mle, 0.3, 0.7)
        self.assertAlmostEqual(0.6749999999999999, dist.prob((START_TOKEN, "aa")))
        self.assertAlmostEqual(0.2833333333333333, dist.prob(("aa", "bb")))


class TestPerplexity(unittest.TestCase):
    @points(2)
    def test_unigram_perplexity_simple(self) -> None:
        """Simple unigram perplexity is correct."""
        counts = {
            "aa": 1,
            "bb": 2,
            "cc": 3,
        }
        mle = UnigramMLE(counts)
        self.assertAlmostEqual(2.0, unigram_perplexity(DefensiveIterable([["cc"]]), mle))

    @points(2)
    def test_bigram_perplexity_simple(self) -> None:
        """Simple bigram perplexity is correct."""
        # Counts for:
        # [["aa", "bb"], ["aa", "bb", "aa"], ["aa", "cc"]]
        counts = {
            START_TOKEN: {"aa": 3},
            "aa": {"bb": 2, END_TOKEN: 1, "cc": 1},
            "bb": {END_TOKEN: 1, "aa": 1},
            "cc": {END_TOKEN: 1},
        }
        mle = BigramMLE(counts)
        self.assertAlmostEqual(
            1.5874010519681994, bigram_perplexity(DefensiveIterable([["aa", "bb"]]), mle)
        )

    @points(2)
    def test_trigram_perplexity_simple(self) -> None:
        """Simple bigram perplexity is correct."""
        # Counts for:
        # [["aa", "bb"], ["aa", "bb", "aa", "bb"], ["aa", "cc"]]
        counts = {
            (START_TOKEN, START_TOKEN): {"aa": 3},
            (START_TOKEN, "aa"): {"bb": 2, "cc": 1},
            ("aa", "bb"): {END_TOKEN: 2, "aa": 1},
            ("bb", "aa"): {"bb": 1},
            ("aa", "cc"): {END_TOKEN: 1},
        }

        mle = TrigramMLE(counts)
        self.assertAlmostEqual(
            1.3103706971044482, trigram_perplexity(DefensiveIterable([["aa", "bb"]]), mle)
        )


def main() -> None:
    tests = [
        TestSampleBigrams,
        TestSampleTrigrams,
        TestMLE,
        TestSequenceProb,
        TestLidstoneSmoothing,
        TestInterpolation,
        TestPerplexity,
    ]
    grader = Grader(tests)
    score = grader.print_results()
    if score == 1.0:
        print("Well done, enjoy this video that goes with the test data:")
        print("https://www.youtube.com/watch?v=eifljYPFW-E")


if __name__ == "__main__":
    main()
