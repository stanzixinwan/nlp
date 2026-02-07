#!/usr/bin/env python

import random
import unittest
from typing import Generator, Iterable, Iterator, Sequence, TypeVar

from grader import Grader, points, timeout
from hw4 import (
    GreedyBigramTagger,
    MostFrequentTagTagger,
    SentenceCounter,
    TaggedToken,
    UnigramTagger,
    ViterbiBigramTagger,
)
from sklearn.metrics import accuracy_score

T = TypeVar("T")


def _make_sentences(sentences: list[list[tuple[str, str]]]) -> list[list[TaggedToken]]:
    return [
        [TaggedToken(token, tag) for token, tag in sentence] for sentence in sentences
    ]


def create_tagged_token(s: str) -> TaggedToken:
    """Create a TaggedToken from a string with the format "token/tag".

    While the tests use this, you do not need to.
    """
    splits = s.rsplit("/", 1)
    assert len(splits) == 2, f"Could not parse token: {repr(s)}"
    return TaggedToken(splits[0], splits[1])


# Has to be defined below make_sentences
SENTENCES_AB_XYZ = _make_sentences(
    [
        [("xx", "AA"), ("xx", "AA"), ("yy", "AA"), ("zz", "AA"), ("zz", "AA")],
        [("xx", "BB"), ("yy", "BB"), ("yy", "BB"), ("yy", "BB"), ("zz", "BB")],
        [("uu", "AA"), ("vv", "BB")],
    ]
)


def load_pos_data(path: str) -> Generator[list[TaggedToken], None, None]:
    """Yield sentences as lists of TaggedTokens."""
    with open(path, encoding="utf8") as infile:
        for line in infile:
            if not line.strip():
                continue
            yield [create_tagged_token(tok) for tok in line.rstrip("\n").split(" ")]


def load_test_pos_data(path: str) -> Iterable[Sequence[TaggedToken]]:
    """Return a defensive iterable over sentences as sequences of tagged tokens."""
    return DefensiveIterable(load_pos_data(path))


class TestMostFrequentTagTagger(unittest.TestCase):
    @points(1)
    def test_most_frequent_tag_sentence(self) -> None:
        """Each token is correctly tagged with the most frequent tag."""
        sentences = _make_sentences([[("foo", "NN"), ("bar", "NNS")], [("baz", "NN")]])
        tagger = MostFrequentTagTagger()
        tagger.train(sentences)

        sentence = ("This", "is", "a", "sentence", ".")
        self.assertListEqual(
            ["NN", "NN", "NN", "NN", "NN"], tagger.tag_sentence(sentence)
        )

    @points(2)
    def test_most_freq_accuracy(self) -> None:
        """Most frequent tagger accuracy is sufficiently high."""
        tagger = MostFrequentTagTagger()
        tagger.train(load_test_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_test_pos_data("test_data/100_dev.txt")
        predicted, actual = tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"MostFrequentTagTagger Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.128, accuracy)


class TestUnigramTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = _make_sentences(
            [
                [("foo", "NN"), ("foo", "NNS"), ("bar", "JJ")],
                [("foo", "NN"), ("bar", "JJ")],
                [("baz", "VB")],
            ]
        )
        self.tagger = UnigramTagger()
        self.tagger.train(self.sentences)

    @points(0.5)
    def test_tag_foo(self) -> None:
        """A single token sentence 'foo' is tagged correctly."""
        tags = self.tagger.tag_sentence(["foo"])
        self.assertEqual("NN", tags[0])

    @points(0.5)
    def test_tag_bar(self) -> None:
        """A single token sentence 'bar' is tagged correctly."""
        tags = self.tagger.tag_sentence(["bar"])
        self.assertEqual("JJ", tags[0])

    @points(1)
    def test_tag_bar_foo(self) -> None:
        """A two-token sentence 'bar foo' is tagged correctly."""
        tags = self.tagger.tag_sentence(["bar", "foo"])
        self.assertListEqual(["JJ", "NN"], tags)

    @points(3)
    def test_unigram_tagger_accuracy(self) -> None:
        """Unigram tagger accuracy is sufficiently high."""
        tagger = UnigramTagger()
        tagger.train(load_test_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_test_pos_data("test_data/100_dev.txt")
        predicted, actual = tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"UnigramTagger Accuracy: {accuracy * 100:.2f}")
        # Solution gets .9297
        self.assertLessEqual(0.9295, accuracy)


class TestInstanceCounterUnsmoothed(unittest.TestCase):
    def setUp(self) -> None:
        self.inst_counter = SentenceCounter(0.0)
        sentences = _make_sentences(
            [
                [("eee", "CC"), ("fff", "AA"), ("eee", "CC")],
                [("hhh", "BB"), ("fff", "CC"), ("ggg", "BB")],
                [("eee", "CC"), ("ggg", "BB")],
            ]
        )
        self.inst_counter.count_sentences(sentences)

    @points(1.5)
    def test_unique_tags(self) -> None:
        """The correctly-sorted tag list is returned."""
        self.assertEqual(["CC", "BB", "AA"], self.inst_counter.unique_tags())

    @points(0.5)
    def test_emission_prob1(self) -> None:
        """Unsmoothed emission probabilities for tag C are correct."""
        self.assertAlmostEqual(3 / 4, self.inst_counter.emission_prob("CC", "eee"))
        self.assertAlmostEqual(1 / 4, self.inst_counter.emission_prob("CC", "fff"))

    @points(0.5)
    def test_emission_prob2(self) -> None:
        """Unsmoothed emission probabilities for tag B are correct."""
        self.assertAlmostEqual(2 / 3, self.inst_counter.emission_prob("BB", "ggg"))
        self.assertAlmostEqual(1 / 3, self.inst_counter.emission_prob("BB", "hhh"))

    @points(0.5)
    def test_emission_prob3(self) -> None:
        """Unsmoothed emission probabilities for tag A are correct."""
        self.assertEqual(1.0, self.inst_counter.emission_prob("AA", "fff"))

    @points(1)
    def test_initial_prob(self) -> None:
        """Observed initial tags have correct initial probabilities."""
        self.assertEqual(2 / 3, self.inst_counter.initial_prob("CC"))
        self.assertEqual(1 / 3, self.inst_counter.initial_prob("BB"))

    @points(0.5)
    def test_transition_prob1(self) -> None:
        """Outgoing transitions have correct probabilities and sum to 1."""
        self.assertEqual(1 / 3, self.inst_counter.transition_prob("CC", "AA"))
        self.assertEqual(2 / 3, self.inst_counter.transition_prob("CC", "BB"))

    @points(0.5)
    def test_transition_prob2(self) -> None:
        """Incoming transitions have correct probabilities and need not sum to 1."""
        self.assertEqual(1.0, self.inst_counter.transition_prob("AA", "CC"))
        self.assertEqual(1.0, self.inst_counter.transition_prob("BB", "CC"))


class TestInstanceCounterSmoothed(unittest.TestCase):
    def setUp(self) -> None:
        self.inst_counter = SentenceCounter(1.0)
        sentences = _make_sentences(
            [
                [("eee", "CC"), ("fff", "AA"), ("eee", "CC")],
                [("hhh", "BB"), ("fff", "CC"), ("ggg", "BB")],
            ]
        )
        self.inst_counter.count_sentences(sentences)

    @points(0.2)
    def test_emission_prob1(self) -> None:
        """Smoothed emission probabilities for tag C are correct."""
        self.assertAlmostEqual(3 / 5, self.inst_counter.emission_prob("CC", "eee"))
        self.assertAlmostEqual(2 / 5, self.inst_counter.emission_prob("CC", "fff"))

    @points(0.2)
    def test_emission_prob2(self) -> None:
        """Smoothed emission probabilities for tag B are correct."""
        self.assertAlmostEqual(2 / 4, self.inst_counter.emission_prob("BB", "ggg"))
        self.assertAlmostEqual(2 / 4, self.inst_counter.emission_prob("BB", "hhh"))

    @points(0.1)
    def test_emission_prob3(self) -> None:
        """Smoothed emission probabilities for tag A are correct."""
        self.assertEqual(1.0, self.inst_counter.emission_prob("AA", "fff"))

    @points(0.5)
    def test_emission_prob4(self) -> None:
        """With smoothing, unobserved emissions have nonzero emission probabilities."""
        self.assertEqual(1 / 2, self.inst_counter.emission_prob("AA", "eee"))
        self.assertEqual(1 / 4, self.inst_counter.emission_prob("BB", "fff"))
        self.assertEqual(1 / 5, self.inst_counter.emission_prob("CC", "ggg"))

    # Initial/transition probabilities are not affected by smoothing, so these tests
    # give minimal points
    @points(0.2)
    def test_initial_prob1(self) -> None:
        """Observed initial tags have correct initial probabilities."""
        self.assertEqual(0.5, self.inst_counter.initial_prob("CC"))
        self.assertEqual(0.5, self.inst_counter.initial_prob("BB"))

    @points(0.2)
    def test_initial_prob2(self) -> None:
        """Unobserved initial tags have initial probability 0."""
        self.assertEqual(0.0, self.inst_counter.initial_prob("AA"))

    @points(0.2)
    def test_transition_prob1(self) -> None:
        """Outgoing transitions have correct probabilities and sum to 1."""
        self.assertEqual(0.5, self.inst_counter.transition_prob("CC", "AA"))
        self.assertEqual(0.5, self.inst_counter.transition_prob("CC", "BB"))

    @points(0.2)
    def test_transition_prob2(self) -> None:
        """Incoming transitions have correct probabilities and need not sum to 1."""
        self.assertEqual(1.0, self.inst_counter.transition_prob("AA", "CC"))
        self.assertEqual(1.0, self.inst_counter.transition_prob("BB", "CC"))

    @points(0.2)
    def test_transition_prob3(self) -> None:
        """Unobserved transitions have probability 0."""
        self.assertEqual(0.0, self.inst_counter.transition_prob("AA", "BB"))
        self.assertEqual(0.0, self.inst_counter.transition_prob("BB", "AA"))


class TestSentenceCounterSpeed(unittest.TestCase):
    @points(7)
    @timeout(3)
    def test_efficient_implementation(self) -> None:
        """Test that SentenceCounter is efficiently implemented.

        If you are failing this test with a TimeoutError, you are not implementing
        SentenceCounter efficiently and are probably using loops or sums in your
        probability functions. Move more computation to count_sentences.
        """
        counter = SentenceCounter(1.0)
        counter.count_sentences(self._make_random_sentences(25_000))
        for _ in range(10000):
            self.assertIsNotNone(counter.unique_tags())
            self.assertIsNotNone(counter.initial_prob("AA"))
            self.assertIsNotNone(counter.transition_prob("AA", "AA"))
            self.assertIsNotNone(counter.emission_prob("AA", "1"))

    @staticmethod
    def _make_random_sentences(
        n_sentences: int,
    ) -> Generator[list[TaggedToken], None, None]:
        random.seed(0)
        tags = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ", "KK"]
        tokens = [str(n) for n in range(n_sentences)]
        lengths = list(range(10, 31))
        for _ in range(n_sentences):
            sen_length = random.choice(lengths)
            sentence = [
                TaggedToken(random.choice(tokens), random.choice(tags))
                for _ in range(sen_length)
            ]
            yield sentence


class TestBigramSequenceProbability(unittest.TestCase):
    def setUp(self) -> None:
        # We test through the greedy tagger but could also do it through Viterbi
        self.tagger = GreedyBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(2)
    def test_prob1(self) -> None:
        """The bigram sequence log-probability is correct."""
        self.assertAlmostEqual(
            -3.518980417318539,
            self.tagger.sequence_probability(["xx", "yy"], ["AA", "AA"]),
        )

    @points(2)
    def test_prob2(self) -> None:
        """The bigram sequence log-probability is correct."""
        self.assertAlmostEqual(
            -3.58351893845611,
            self.tagger.sequence_probability(["xx", "yy"], ["BB", "BB"]),
        )


class TestGreedyBigramTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.tagger = GreedyBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(0.5)
    def test_ab_xyz_tag1(self) -> None:
        """The greedy tagger correctly tags 'xx xx'."""
        sent = ["xx", "xx"]
        self.assertEqual(["AA", "AA"], self.tagger.tag_sentence(sent))

    @points(0.5)
    def test_ab_xyz_tag2(self) -> None:
        """The greedy tagger correctly tags 'yy yy'."""
        sent = ["yy", "yy"]
        self.assertEqual(["BB", "BB"], self.tagger.tag_sentence(sent))

    @points(0.5)
    def test_ab_xyz_tag3(self) -> None:
        """The greedy tagger correctly tags 'xx yy'."""
        sent = ["xx", "yy"]
        self.assertEqual(["AA", "AA"], self.tagger.tag_sentence(sent))

    @points(0.5)
    def test_ab_xyz_tag4(self) -> None:
        """The greedy tagger correctly tags 'xx zz'."""
        sent = ["xx", "zz"]
        self.assertEqual(["AA", "AA"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag5(self) -> None:
        """The Viterbi tagger correctly tags 'uu vv'."""
        sent = ["uu", "vv"]
        self.assertEqual(["AA", "BB"], self.tagger.tag_sentence(sent))

    @points(5)
    def test_greedy_tagger_accuracy(self) -> None:
        """Accuracy is sufficiently high."""
        tagger = GreedyBigramTagger(0.001)
        tagger.train(load_test_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_test_pos_data("test_data/100_dev.txt")
        predicted, actual = tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"GreedyBigram Accuracy: {accuracy * 100:.2f}")
        # Solution gets .9536
        self.assertLessEqual(0.9534, accuracy)


class TestViterbiBigramTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.tagger = ViterbiBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(1)
    def test_ab_xyz_tag1(self) -> None:
        """The Viterbi tagger correctly tags 'xx xx'."""
        sent = ["xx", "xx"]
        self.assertEqual(["AA", "AA"], self.tagger.tag_sentence(sent))

    @points(1)
    def test_ab_xyz_tag2(self) -> None:
        """The Viterbi tagger correctly tags 'yy yy'."""
        sent = ["yy", "yy"]
        self.assertEqual(["BB", "BB"], self.tagger.tag_sentence(sent))

    @points(1)
    def test_ab_xyz_tag3(self) -> None:
        """The Viterbi tagger correctly tags 'xx yy'."""
        sent = ["xx", "yy"]
        self.assertEqual(["AA", "AA"], self.tagger.tag_sentence(sent))

    @points(1)
    def test_ab_xyz_tag4(self) -> None:
        """The Viterbi tagger correctly tags 'xx zz'."""
        sent = ["xx", "zz"]
        self.assertEqual(["AA", "AA"], self.tagger.tag_sentence(sent))

    @points(1)
    def test_ab_xyz_tag5(self) -> None:
        """The Viterbi tagger correctly tags 'uu vv'."""
        sent = ["uu", "vv"]
        self.assertEqual(["AA", "BB"], self.tagger.tag_sentence(sent))

    @points(10)
    def test_viterbi_tagger_accuracy(self) -> None:
        tagger = ViterbiBigramTagger(0.001)
        tagger.train(load_test_pos_data("test_data/train_pos.txt"))
        # Test on smaller dev set for speed purposes
        tagged_sents = load_test_pos_data("test_data/100_dev.txt")
        predicted, actual = tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"ViterbiBigram Accuracy: {accuracy * 100:.2f}")
        # Solution gets .9624
        self.assertLessEqual(0.9622, accuracy)


class DefensiveIterable(Iterable[T]):
    def __init__(self, source: Iterable[T]) -> None:
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


def main() -> None:
    tests = [
        TestMostFrequentTagTagger,
        TestUnigramTagger,
        TestInstanceCounterUnsmoothed,
        TestInstanceCounterSmoothed,
        TestSentenceCounterSpeed,
        TestBigramSequenceProbability,
        TestGreedyBigramTagger,
        TestViterbiBigramTagger,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
