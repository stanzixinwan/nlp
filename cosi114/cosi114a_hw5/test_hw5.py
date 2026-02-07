#!/usr/bin/env python

import unittest
from pathlib import Path
from typing import Counter, Sequence

import numpy as np
from grader import Grader, points
from hw5 import (
    CharBigramFeatureExtractor,
    CharTrigramFeatureExtractor,
    CharUnigramFeatureExtractor,
    ClassificationInstance,
    LanguageIdentificationInstance,
    MulticlassScoring,
    MultinomialLogisticRegression,
    Vectorizer,
    highest_prob_index,
    load_lid_instances,
)
from sklearn.metrics import accuracy_score

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"

TEST_DATA_PATH = Path("test_data")
LID_TRAIN_PATH = TEST_DATA_PATH / "mot_train.tsv"
LID_DEV_PATH = TEST_DATA_PATH / "mot_dev.tsv"
# We store this is a tuple to prevent accidental modification, but
# it needs to be converted to a list before training.
SENTIMENT_DATA = (
    ClassificationInstance(
        "positive",
        ["I", "love", "tacos", "!"],
    ),
    ClassificationInstance(
        "negative",
        ["I", "dislike", "broccoli", "."],
    ),
    ClassificationInstance(
        "negative",
        [
            "I",
            "love",
            "to",
            "dislike",
            "tacos",
        ],
    ),
)


class TestScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.predicted = [
            "positive",
            "positive",
            "neutral",
            "positive",
            "negative",
            "negative",
            "positive",
            "neutral",
            "positive",
            "negative",
        ]
        self.true = [
            "neutral",
            "positive",
            "positive",
            "neutral",
            "negative",
            "negative",
            "positive",
            "negative",
            "negative",
            "negative",
        ]
        self.scorer = MulticlassScoring(["positive", "neutral", "negative"])
        self.scorer.score(self.true, self.predicted)

    @points(2)
    def test_accuracy(self) -> None:
        """Accuracy is correct for a simple case."""
        self.assertEqual(float, type(self.scorer.accuracy()))
        self.assertAlmostEqual(
            accuracy_score(self.true, self.predicted), self.scorer.accuracy()
        )

    @points(0.5)
    def test_macro_f1_type(self) -> None:
        """Macro F1 returns a float."""
        self.assertEqual(float, type(self.scorer.macro_f1()))

    @points(0.5)
    def test_weighted_f1_type(self) -> None:
        """Weighted F1 returns a float."""
        self.assertEqual(float, type(self.scorer.weighted_f1()))

    @points(3)
    def test_confusion_counts(self) -> None:
        """Confusion counts are correct for a simple case."""
        self.assertEqual(1, self.scorer.confusion_count("positive", "neutral"))
        self.assertEqual(2, self.scorer.confusion_count("positive", "positive"))

    @points(3)
    def test_confusion_rates(self) -> None:
        """Confusion rates are correct for a simple case."""
        self.assertAlmostEqual(1 / 3, self.scorer.confusion_rate("positive", "neutral"))
        self.assertAlmostEqual(0.0, self.scorer.confusion_rate("positive", "negative"))


class TestFeatureExtraction(unittest.TestCase):
    @points(3)
    def test_unigram_features(self) -> None:
        """The correct set of unigram features is generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hi!")
        features = set(
            CharUnigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"h", "i", "!"},
            features,
        )

    @points(3)
    def test_bigram_features(self) -> None:
        """The correct set of bigram features is generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hello!")
        features = set(
            CharBigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"lo", "el", "he", "o!", "ll"},
            features,
        )

    @points(3)
    def test_trigram_features(self) -> None:
        """The correct set of trigram features is generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hello!")
        features = set(
            CharTrigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"hel", "ell", "llo", "lo!"},
            features,
        )


class TestVectorizer(unittest.TestCase):
    DATA = [
        ClassificationInstance("A", {"a": 1}),
        ClassificationInstance("B", {"b": 1}),
        ClassificationInstance("A", {"a": 1, "b": 1}),
    ]

    def setUp(self) -> None:
        self.vect = Vectorizer()
        self.vect.learn_features_and_labels(self.DATA)

    @points(2)
    def test_single_feature(self) -> None:
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        self.assertTrue(
            np.array_equal(expected, self.vect.features_to_vectors(self.DATA))
        )

    @points(1)
    def test_labels_to_indices(self) -> None:
        expected = np.array([0, 1, 0], dtype=np.int64)
        self.assertTrue(np.array_equal(expected, self.vect.labels_to_indices(self.DATA)))

    @points(1)
    def test_indices_to_vectors(self) -> None:
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        indices = self.vect.labels_to_indices(self.DATA)
        self.assertTrue(
            np.array_equal(expected, self.vect.label_indices_to_vectors(indices))
        )

    @points(1)
    def test_indices_to_labels(self) -> None:
        expected = ["A", "B", "A"]
        label_indices = self.vect.labels_to_indices(self.DATA)
        self.assertEqual(expected, self.vect.label_indices_to_labels(label_indices))


class TestModelPredictions(unittest.TestCase):
    EVAL_DATA_PATH = LID_DEV_PATH
    train_instances: list[ClassificationInstance]
    eval_instances: list[ClassificationInstance]
    eval_labels: list[str]
    labels: list[str]
    vectorizer: Vectorizer
    feature_extractor_not_implemented: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        feature_extractor = CharBigramFeatureExtractor()
        cls.train_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_lid_instances(LID_TRAIN_PATH, 20_000)
        ]
        # Crash now if the feature extractor isn't implemented
        if cls.train_instances[0] is None:
            cls.feature_extractor_not_implemented = True
            return

        cls.eval_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_lid_instances(cls.EVAL_DATA_PATH)
        ]
        cls.eval_labels = [instance.label for instance in cls.eval_instances]
        cls.labels = sorted(Counter(instance.label for instance in cls.train_instances))
        cls.vectorizer = Vectorizer()

    def setUp(self) -> None:
        if self.feature_extractor_not_implemented:
            raise ValueError(
                "Cannot test model predictions without an implemented feature extractor"
            )

    @points(5)
    def test_train_no_regularization(self) -> None:
        dev_acc, final_train_acc = train_eval_model(
            self.train_instances,
            self.eval_instances,
            self.vectorizer,
            n_iter=250,
            lr=2e-3,
            l2=0.0,
        )
        # Check training accuracy first, should be about 0.99
        self.assertGreater(final_train_acc, 0.985)

        # Solution gets 0.9223
        # Upper bound for accuracy
        self.assertLess(dev_acc, 0.9233)
        # Lower bound for accuracy
        self.assertGreater(dev_acc, 0.9123)

    @points(2)
    def test_train_with_regularization(self) -> None:
        dev_acc, final_train_acc = train_eval_model(
            self.train_instances,
            self.eval_instances,
            self.vectorizer,
            n_iter=250,
            lr=2e-3,
            l2=1e-3,
        )
        # Check training accuracy first, should be about 0.99
        self.assertGreater(final_train_acc, 0.985)

        # Solution gets 0.9240
        # Upper bound for accuracy
        self.assertLess(dev_acc, 0.925)
        # Lower bound for accuracy
        self.assertGreater(dev_acc, 0.923)


def train_eval_model(
    train_instances: Sequence[ClassificationInstance],
    dev_instances: Sequence[ClassificationInstance],
    vectorizer: Vectorizer,
    n_iter: int,
    lr: float,
    l2: float,
) -> tuple[float, float]:
    vectorizer.learn_features_and_labels(train_instances)
    train_x = vectorizer.features_to_vectors(train_instances)
    train_y = vectorizer.labels_to_indices(train_instances)

    model = MultinomialLogisticRegression(vectorizer)
    print(
        f"Training for {n_iter} epochs on {len(train_y)} instances with "
        f"lr of {lr} and l2 coef of {l2}, be patient..."
    )
    train_accs = model.train(train_x, train_y, n_iter=n_iter, lr=lr, l2_coeff=l2)
    final_acc = train_accs[-1]
    print("Final training accuracy:", final_acc)

    dev_x = vectorizer.features_to_vectors(dev_instances)
    dev_y = vectorizer.labels_to_indices(dev_instances)
    dev_y_pred = highest_prob_index(model.predict(dev_x))
    acc = accuracy_score(dev_y, dev_y_pred)
    print("Eval accuracy:", acc)
    return acc, final_acc


def main() -> None:
    tests = [
        TestScoring,
        TestFeatureExtraction,
        TestVectorizer,
        # Should be last since it's slowest
        TestModelPredictions,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
