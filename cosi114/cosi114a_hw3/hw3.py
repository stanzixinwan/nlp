from collections import Counter, defaultdict
from math import log
from typing import (
    Iterable,
    Sequence,
)
# Helper imports for set loading
import json
import os
from typing import Generator, Union
from pathlib import Path
# Version 1.0.0
# 10/11/2024

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
        self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
        self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.

def true_positive(predictions: Sequence[str], expected: Sequence[str], positive_label: str) -> int:
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == expected[i] and expected[i] == positive_label:
            count += 1
    return count

def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError("Neither predictions nor expected can be empty")
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected have different length")
    return sum(e == p for e, p in zip(expected, predictions)) / len(predictions)

def recall(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> float:
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError("Neither predictions nor expected can be empty")
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected have different length")

    expected_positive = sum(positive_label == e for e in expected)
    if expected_positive == 0:
        return 0.0
    return true_positive(predictions, expected, positive_label) / expected_positive

def precision(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> float:
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError("Neither predictions nor expected can be empty")
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected have different length")

    predicted_positive = sum(positive_label == p for p in predictions)
    if predicted_positive == 0:
        return 0.0
    return true_positive(predictions, expected, positive_label) / predicted_positive


def f1(predictions: Sequence[str], expected: Sequence[str], positive_label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    pre = precision(predictions, expected, positive_label)
    rec = recall(predictions, expected, positive_label)
    if pre + rec == 0:
        return 0.0
    return 2 * (pre * rec) / (pre + rec)


class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        features_set = set()
        for sentence in instance.sentences:
            for token in sentence:
                features_set.add(token.lower())
        return ClassificationInstance(instance.label, features_set)

def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    bigram_tuples = [('<START>',sentence[0])]
    for i in range(len(sentence)-1):
        bigram_tuples.append((sentence[i],sentence[i+1]))
    bigram_tuples.append((sentence[len(sentence)-1],'<END>'))
    return bigram_tuples

class BigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        features_set = set()
        for sentence in instance.sentences:
            lowercased_sentence = []
            for token in sentence:
                lowercased_sentence.append(token.lower())
            for bigram in bigrams(lowercased_sentence):
                features_set.add(str(bigram))
        return ClassificationInstance(instance.label, features_set)


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        features_list = [str('left_tok=' + instance.left_context),
                         str('split_tok=' + instance.token),
                         str('right_tok=' + instance.right_context)]
        return ClassificationInstance(instance.label, features_list)


class InstanceCounter:
    def __init__(self) -> None:
        self.label_counts = Counter()
        self.total_label_counts = 0
        self.feature_label_count_dict = {}
        self.label_list = []
        self.feature_vocabs = set()
        self.total_feature_count = Counter()

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        uniq_labels = set()
        for instance in instances:
            self.label_counts[instance.label] += 1
            self.total_label_counts += 1
            if instance.label not in self.feature_label_count_dict:
                self.feature_label_count_dict[instance.label] = Counter()
            for feature in instance.features:
                self.feature_label_count_dict[instance.label][feature] += 1
                self.feature_vocabs.add(feature)
                self.total_feature_count[instance.label] += 1
            uniq_labels.add(instance.label)
        self.label_list = list(uniq_labels)

    def label_count(self, label: str) -> int:
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.total_label_counts

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        return self.feature_label_count_dict[label][feature]

    def unique_labels(self) -> list[str]:
        return self.label_list

    def feature_vocab_size(self) -> int:
        return len(self.feature_vocabs)

    def feature_set(self) -> set[str]:
        return self.feature_vocabs

    def total_feature_count_for_label(self, label: str) -> int:
        return self.total_feature_count[label]


class NaiveBayesClassifier:
    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        return self.instance_counter.label_count(label) / self.instance_counter.total_label_counts

    def feature_prob(self, feature: str, label) -> float:
        if feature not in self.instance_counter.feature_set():
            return 0.0
        return (self.instance_counter.feature_label_joint_count(feature, label) + self.k) / (self.instance_counter.total_feature_count_for_label(label) + self.k * self.instance_counter.feature_vocab_size()) # c+k/N+k*V

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        log_posterior_prob = log(self.prior_prob(label))
        for feature in features:
            if feature in self.instance_counter.feature_set():
                log_posterior_prob += log(self.feature_prob(feature, label))
        return log_posterior_prob

    def classify(self, features: Sequence[str]) -> str:
        probs = []
        for label in self.instance_counter.unique_labels():
            probs.append((self.log_posterior_prob(features, label), label))
        return max(probs)[1]

    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        predictions = []
        true_labels = []
        for instance in instances:
            predictions.append(self.classify(instance.features))
            true_labels.append(instance.label)
        return predictions, true_labels


# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = float(0.5)
# This configuration has the highest Accuracy: 93.76

def classification_report(predicted: Sequence[str], expected: Sequence[str], positive_label: str) -> tuple[float, float, float, float, str]:
    """Return accuracy, P, R, F1 and a classification report."""
    acc = accuracy(predicted, expected)
    prec = precision(predicted, expected, positive_label)
    rec = recall(predicted, expected, positive_label)
    f1_ = f1(predicted, expected, positive_label)
    report = "\n".join(
        [
            f"Accuracy:  {acc * 100:0.2f}",
            f"Precision: {prec * 100:0.2f}",
            f"Recall:    {rec * 100:0.2f}",
            f"F1:        {f1_ * 100:0.2f}",
        ]
    )
    return acc, prec, rec, f1_, report

def load_sentiment_instances(
    datapath: Union[str, Path],
) -> Generator[AirlineSentimentInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_dict in json_list:
            yield AirlineSentimentInstance(
                json_dict["label"], json_dict["airline"], json_dict["sentences"]
            )

def grid_search() -> None:

    k_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    raw_train_instances = list(load_sentiment_instances(os.path.join(Path("test_data") / "airline_sentiment", "train.json")))
    raw_dev_instances = list(load_sentiment_instances(os.path.join(Path("test_data") / "airline_sentiment", "dev.json")))

    for k in k_values:
        extractor = TunedAirlineSentimentFeatureExtractor()
        extractor.k = k
        sentiment_train_instances = [extractor.extract_features(inst) for inst in raw_train_instances]

        sentiment_dev_instances = [extractor.extract_features(inst) for inst in raw_dev_instances]
        sentiment_classifier = NaiveBayesClassifier(extractor.k)
        sentiment_classifier.train(sentiment_train_instances)
        predicted, expected = sentiment_classifier.test(sentiment_dev_instances)
        for positive_label in sentiment_classifier.instance_counter.unique_labels():
            _, _, _, _, report = classification_report(
                predicted, expected, positive_label
            )
            print(
                f"Tuned sentiment classification performance for k of "
                f"{extractor.k} with positive label {repr(positive_label)}:"
            )
            print(report)
            print()

if __name__ == "__main__":
    grid_search()