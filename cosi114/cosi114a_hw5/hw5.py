from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Generator, Iterable, Sequence, Union

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        # We have to sort to ensure determinism, even if it's slow
        self.features: tuple[str, ...] = tuple(sorted(features))
        assert self.features, f"Empty features: {features}"
        for feature in self.features:
            assert isinstance(feature, str), f"Non-string feature: {repr(feature)}"

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


# DO NOT MODIFY
class LanguageIdentificationInstance:
    """Represent a single instance from a language ID dataset."""

    def __init__(
        self,
        language: str,
        text: str,
    ) -> None:
        self.language: str = language
        self.text: str = text

    def __repr__(self) -> str:
        return f"<LanguageIdentificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.language}; text={self.text}"

    # DO NOT CALL THIS METHOD
    # It's called by data loading functions.
    @classmethod
    def from_line(cls, line: str) -> "LanguageIdentificationInstance":
        splits = line.rstrip("\n").split("\t")
        assert len(splits) == 2
        assert splits[0]
        assert len(splits[1]) >= 2, f"Line too short: {repr(line)}"
        return cls(splits[0], splits[1])


# DO NOT MODIFY
def load_lid_instances(
    path: Union[str, Path],
    max_instances: int = 0,
) -> Generator[LanguageIdentificationInstance, None, None]:
    """Load LID instances from a file."""
    count = 0
    with open(path, encoding="utf8") as file:
        for line in file:
            yield LanguageIdentificationInstance.from_line(line)
            count += 1
            if max_instances and count == max_instances:
                break


# DO NOT MODIFY
class FeatureExtractor(ABC):
    """An abstract class for representing feature extractors."""

    @staticmethod
    @abstractmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        raise NotImplementedError


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MulticlassScoring:
    def __init__(self, labels: Sequence[str]) -> None:
        self.labels: Sequence[str] = labels
        self.confusion_matrix: dict[tuple[str, str], int] = defaultdict(int)
        self.true_label_counts: dict[str, int] = defaultdict(int)
        self.predicted_label_counts: dict[str, int] = defaultdict(int)
        self.total_correct: int = 0
        self.total_count: int = 0

    def score(
        self,
        true_labels: Sequence[str],
        predicted_labels: Sequence[str],
    ) -> None:
        if len(true_labels) != len(predicted_labels):
            raise ValueError("Number of labels does not match number of predicted labels")
        if len(true_labels)==0 or len(predicted_labels)==0:
            raise ValueError("Empty labels")

        self.total_count = len(true_labels)
        for i in range(len(true_labels)):
            true_label = true_labels[i]
            predicted_label = predicted_labels[i]
            self.confusion_matrix[(true_label, predicted_label)] += 1
            self.true_label_counts[true_label] += 1
            self.predicted_label_counts[predicted_label] += 1
            if true_label == predicted_label:
                self.total_correct += 1

    def accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_correct / self.total_count

    def precision(self, label: str) -> float:
        tp = self.confusion_matrix.get((label, label), 0)
        denominator = self.predicted_label_counts.get(label, 0) # TP + FP
        if denominator == 0:
            return 0.0
        return tp / denominator

    def recall(self, label: str) -> float:
        tp = self.confusion_matrix.get((label, label), 0)
        denominator = self.true_label_counts.get(label, 0) # TP + FN
        if denominator == 0:
            return 0.0
        return tp / denominator

    def f1(self, label: str) -> float:
        prec = self.precision(label)
        rec = self.recall(label)
        denominator = prec + rec
        if denominator == 0:
            return 0.0
        return 2 * prec * rec / denominator

    def macro_f1(self) -> float:
        f1_scores = [self.f1(label) for label in self.labels]
        return sum(f1_scores) / len(f1_scores)

    def weighted_f1(self) -> float:
        weighted_sum = 0.0
        for label in self.labels:
            weight = self.true_label_counts.get(label, 0) / self.total_count
            weighted_sum += weight * self.f1(label)
        return weighted_sum

    def confusion_count(self, true_label: str, predicted_label: str) -> int:
        return self.confusion_matrix.get((true_label, predicted_label), 0)

    def confusion_rate(self, true_label: str, predicted_label: str) -> float:
        count = self.confusion_count(true_label, predicted_label)
        true_count = self.true_label_counts.get(true_label, 0)
        if true_count == 0:
            return 0.0
        return count / true_count


class CharUnigramFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        unique_chars = set(instance.text)
        return ClassificationInstance(instance.language, unique_chars)

class CharBigramFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        text = instance.text
        bigrams = set()
        for i in range(len(text) - 1):
            bigrams.add(text[i:i + 2])
        return ClassificationInstance(instance.language, bigrams)


class CharTrigramFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        text = instance.text
        trigrams = set()
        for i in range(len(text) - 2):
            trigrams.add(text[i:i + 3])
        return ClassificationInstance(instance.language, trigrams)


class Vectorizer:
    def __init__(self) -> None:
        self.feature_to_index: dict[str, int] = {}
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: dict[int, str] = {}

    def learn_features_and_labels(self, data: Sequence[ClassificationInstance]) -> None:
        # Learn features and labels in order of first appearance
        for instance in data:
            # Learn label
            if instance.label not in self.label_to_index:
                index = len(self.label_to_index)
                self.label_to_index[instance.label] = index
                self.index_to_label[index] = instance.label

            # Learn features
            for feature in instance.features:
                if feature not in self.feature_to_index:
                    self.feature_to_index[feature] = len(self.feature_to_index)

    def n_labels(self) -> int:
        return len(self.label_to_index)

    def unique_labels(self) -> list[str]:
        return list(self.label_to_index.keys())

    def n_features(self) -> int:
        return len(self.feature_to_index)

    def features_to_vectors(
        self, data: Sequence[ClassificationInstance]
    ) -> npt.NDArray[np.float64]:
        n = len(data)
        m = len(self.feature_to_index)
        matrix = np.zeros((n, m), dtype=np.float64)

        for i, instance in enumerate(data):
            for feature in instance.features:
                if feature in self.feature_to_index:
                    j = self.feature_to_index[feature]
                    matrix[i, j] = 1.0

        return matrix

    def labels_to_indices(
        self, data: Sequence[ClassificationInstance]
    ) -> npt.NDArray[np.int64]:
        indices = []
        for instance in data:
            indices.append(self.label_to_index[instance.label])
        return np.array(indices, dtype=np.int64)

    def label_indices_to_vectors(
        self,
        label_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        n = len(label_indices)
        c = len(self.label_to_index)
        matrix = np.zeros((n, c), dtype=np.float64)
        # Set the appropriate indices to 1.0
        for i, label_idx in enumerate(label_indices):
            matrix[i, label_idx] = 1.0
        return matrix

    def label_indices_to_labels(self, y: npt.NDArray[np.int64]) -> list[str]:
        return [self.index_to_label[idx] for idx in y]


def softmax(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    z_max = np.max(z, axis=1, keepdims=True)
    z_shifted = z - z_max
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / sum_exp


def highest_prob_index(z: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.argmax(z, axis=1).astype(np.int64)


class MultinomialLogisticRegression:
    def __init__(self, vectorizer: Vectorizer) -> None:
        self.vectorizer = vectorizer
        f = vectorizer.n_features()
        c = vectorizer.n_labels()
        # Weights: f x c matrix
        self.weights = np.zeros((f, c), dtype=np.float64)
        # Bias: length c array
        self.bias = np.zeros(c, dtype=np.float64)

    def train(
        self,
        train_x: npt.NDArray[np.float64],
        train_y: npt.NDArray[np.int64],
        lr: float,
        l2_coeff: float,
        n_iter: int,
    ) -> list[float]:

        accuracies = []
        # Convert label indices to one-hot vectors
        train_y_onehot = self.vectorizer.label_indices_to_vectors(train_y)

        for epoch in range(n_iter):

            probs = self.predict(train_x)

            predictions = highest_prob_index(probs)
            acc = accuracy_score(train_y, predictions)
            accuracies.append(acc)

            # Error: predicted - true probabilities
            error = probs - train_y_onehot  # shape: n x c

            # Gradient for weights: X^T @ error
            weight_gradient = train_x.T @ error # (shape: f x c)

            # Gradient for bias: sum of errors across all data points
            bias_gradient = np.sum(error, axis=0) # (shape: c)

            # For each weight w: -sign(w) * w^2 * l2_coeff
            inverse_weight_sign = np.sign(self.weights)
            weight_squared = self.weights ** 2
            l2_regularization = inverse_weight_sign * weight_squared * l2_coeff

            # Weights: subtract lr * gradient and subtract L2 regularization
            self.weights = self.weights - lr * weight_gradient - l2_regularization
            # Bias: subtract lr * gradient
            self.bias = self.bias - lr * bias_gradient

        return accuracies

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: n x f, weights: f x c, result: n x c
        scores = x @ self.weights + self.bias
        return softmax(scores)


if __name__ == "__main__":
    pass