#! /usr/bin/env python

import os
import unittest
from types import GeneratorType

from grader import Grader, points
from hw0 import case_sarcastically, gen_sentences, n_most_frequent_tokens, sorted_chars


class TestGenSentences(unittest.TestCase):
    @points(3)
    def test_type(self) -> None:
        """A generator is returned."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_1.txt"))
        self.assertEqual(GeneratorType, type(gen))

    @points(5)
    def test_basic(self) -> None:
        """A basic file is read correctly."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_1.txt"))
        self.assertEqual(
            ["Tokenized", "text", "is", "easy", "to", "work", "with", "."], next(gen)
        )
        self.assertEqual(["Writing", "a", "tokenizer", "is", "a", "pain", "."], next(gen))
        with self.assertRaises(StopIteration):
            next(gen)


class TestSarcasticCaser(unittest.TestCase):
    @points(3)
    def test_no_punc(self) -> None:
        """Casing is correct for letters."""
        self.assertEqual("hElLo", case_sarcastically("hello"))

    @points(2)
    def test_punc1(self) -> None:
        """Casing is correct with punctuation."""
        self.assertEqual("hElLo, FrIeNd!", case_sarcastically("hello, friend!"))

    @points(2)
    def test_punc2(self) -> None:
        """Casing is correct with punctuation."""
        self.assertEqual(
            'sAy "HeLlO," fRiEnD‽', case_sarcastically('Say "hello," friend‽')
        )


class TestSortedChars(unittest.TestCase):
    @points(5)
    def test_sorted_chars_unique(self) -> None:
        """Each letter only appears once."""
        self.assertEqual(["a", "b", "d", "e"], sorted_chars("abbbddeeeee"))


class TestNMostFrequentWords(unittest.TestCase):
    @points(5)
    def test_n_most_frequent_words(self) -> None:
        """The most frequent words are returned in a simple case."""
        sentences = [
            ["a", "b", "c"],
            ["a", "b"],
            ["a"],
        ]
        self.assertEqual(["a", "b"], n_most_frequent_tokens(sentences, 2))


def main() -> None:
    tests = [
        TestGenSentences,
        TestSarcasticCaser,
        TestSortedChars,
        TestNMostFrequentWords,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
