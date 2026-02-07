#! /usr/bin/env python

import os
import sys
import unittest

from grader import Grader, points
from hw0 import case_sarcastically, gen_sentences, n_most_frequent_tokens, sorted_chars
from test_hw0 import (
    TestGenSentences,
    TestNMostFrequentWords,
    TestSarcasticCaser,
    TestSortedChars,
)


class GradeGenSentences(unittest.TestCase):
    @points(3)
    def test_blank_line(self) -> None:
        """Blank lines are skipped."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_empty_line.txt")
        )
        self.assertEqual(["Hello", ",", "world", "!"], next(gen))
        # Between these sentences, there is an empty line in the file which should be skipped over.
        self.assertEqual(["This", "is", "a", "normal", "sentence", "."], next(gen))
        self.assertEqual(
            [
                '"',
                "I",
                "don't",
                "like",
                "it",
                "when",
                "there's",
                "too",
                "much",
                "punctuation",
                "!",
                '"',
                ",",
                "they",
                "exclaimed",
                ".",
            ],
            next(gen),
        )
        self.assertEqual([":)"], next(gen))
        with self.assertRaises(StopIteration):
            next(gen)

    @points(2)
    def test_whitespace_token1(self) -> None:
        """Any character other than space or newline can be a token (or part of a token)."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_whitespace_tokens1.txt")
        )
        self.assertEqual(["tab", "\t", "can", "be", "a", "token"], next(gen))

    @points(2)
    def test_whitespace_token2(self) -> None:
        """Any character other than space or newline can be a token (or part of a token)."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_whitespace_tokens2.txt")
        )
        self.assertEqual(
            ["lines", "can", "end", "in", "a", "tab", "token", "\t"], next(gen)
        )

    @points(2)
    def test_no_newline_at_end_of_line(self) -> None:
        """Not all lines end in newline."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_no_newline.txt")
        )
        self.assertEqual(["not", "all", "lines", "end", "in", "newline"], next(gen))

    @points(3)
    def test_unicode(self) -> None:
        """Unicode files are opened correctly."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_unicode.txt")
        )
        self.assertEqual(
            [
                "ሕድሕድ",
                "ሰብ",
                "ብህይወት",
                "ናይ",
                "ምንባር",
                "፣",
                "ናይ",
                "ነፃነትን",
                "ድሕንነትን",
                "መሰል",
                "ኦለዎ",
                "፡፡",
            ],
            next(gen),
        )


class GradeSarcasticCaser(unittest.TestCase):
    @points(2)
    def test_caseless_letters(self) -> None:
        """Scripts without casing are not affected."""
        # From the UDHR in Tigrinya, with some Latin characters thrown in
        self.assertEqual(
            "a ሕድሕድ ሰብ ብህይወት ናይ ምንባር፣ A ናይ ነፃነትን ድሕንነትን መሰል ኦለዎ፡፡ a",
            case_sarcastically("a ሕድሕድ ሰብ ብህይወት ናይ ምንባር፣ a ናይ ነፃነትን ድሕንነትን መሰል ኦለዎ፡፡ a"),
        )

    @points(1)
    def test_punc1(self) -> None:
        """Non-roman scripts and unusual whitespace are processed correctly."""
        self.assertEqual("ѣ Α γ Δ　ӭ", case_sarcastically("Ѣ α Γ δ　Ӭ"))

    @points(1)
    def test_punc2(self) -> None:
        """Unusual Unicode characters are processed correctly."""
        self.assertEqual("a☞A◆a☹", case_sarcastically("a☞a◆a☹"))


class GradeSortedChars(unittest.TestCase):
    @points(2)
    def test_sorted_chars_sorted(self) -> None:
        """Characters appear in sorted order."""
        self.assertEqual(["a", "b", "c", "d"], sorted_chars("ccaabbabacbdbcba"))

    @points(1)
    def test_sorted_chars_empty(self) -> None:
        """An empty string results in an empty list."""
        self.assertEqual([], sorted_chars(""))


class GradeNMostFrequentWords(unittest.TestCase):
    SENTENCES1 = [
        ["the", "cat"],
        ["the", "dog"],
        ["the", "cat"],
    ]

    @points(3)
    def test_negative_n(self) -> None:
        """N cannot be negative."""
        with self.assertRaises(ValueError):
            n_most_frequent_tokens(self.SENTENCES1, -1)

    @points(1)
    def test_zero_n(self) -> None:
        """N can be zero."""
        self.assertEqual([], n_most_frequent_tokens(self.SENTENCES1, 0))

    @points(1)
    def test_empty_sentences(self) -> None:
        """Sentences can be empty."""
        self.assertEqual([], n_most_frequent_tokens([], 1))

    @points(1)
    def test_large_n(self) -> None:
        """N can be larger than the vocabulary."""
        self.assertEqual(
            ["the", "cat", "dog"], n_most_frequent_tokens(self.SENTENCES1, 100)
        )


def main() -> None:
    tests = [
        TestGenSentences,
        TestSarcasticCaser,
        TestSortedChars,
        TestNMostFrequentWords,
        GradeGenSentences,
        GradeSarcasticCaser,
        GradeSortedChars,
        GradeNMostFrequentWords,
    ]
    grader = Grader(tests)
    grader.print_results(sys.stderr)


if __name__ == "__main__":
    main()
