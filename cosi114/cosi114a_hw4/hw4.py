from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import log
from operator import itemgetter
from typing import Generator, Iterable, Sequence

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.
# HW 4 stubs 1.0.0 10/21/2024

# DO NOT MODIFY
NEG_INF = float("-inf")


# DO NOT MODIFY
class TaggedToken:
    """Store the text and tag for a token."""

    # DO NOT MODIFY
    def __init__(self, text: str, tag: str) -> None:
        self.text: str = text
        self.tag: str = tag

    # DO NOT MODIFY
    def __str__(self) -> str:
        return f"{self.text}/{self.tag}"

    # DO NOT MODIFY
    def __repr__(self) -> str:
        return f"<TaggedToken {str(self)}>"

    # DO NOT MODIFY
    def __hash__(self) -> int:
        raise ValueError(
            "Do not try to put TaggedToken objects in a dictionary or set. "
            "You probably want the text or tag attribute rather than the whole object."
        )


# DO NOT MODIFY
class Tagger(ABC):
    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """Tag a sentence with part of speech tags."""
        raise NotImplementedError

    # DO NOT MODIFY
    def tag_sentences(
        self, sentences: Iterable[Sequence[str]]
    ) -> Generator[list[str], None, None]:
        """Yield a list of tags for each sentence in the input."""
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    # DO NOT MODIFY
    def test(
        self, tagged_sentences: Iterable[Sequence[TaggedToken]]
    ) -> tuple[list[str], list[str]]:
        """Return a tuple containing a list of predicted tags and a list of actual tags.

        Does not preserve sentence boundaries to make evaluation simpler.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sentence in tagged_sentences:
            predicted.extend(self.tag_sentence([tok.text for tok in sentence]))
            actual.extend([tok.tag for tok in sentence])
        return predicted, actual


# DO NOT MODIFY
def safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter.

    In case of ties, the lexicographically first item is returned.
    """
    assert counts, "Counter is empty"
    return items_descending_value(counts)[0]


# DO NOT MODIFY
def items_descending_value(counts: Counter[str]) -> list[str]:
    """Return the keys in descending frequency, breaking ties lexicographically."""
    # Why can't we just use most_common? It sorts by descending frequency, but items
    # of the same frequency follow insertion order, which we can't depend on.
    # Why can't we just use sorted with reverse=True? It will give us descending
    # by count, but reverse lexicographic sorting, which is confusing.
    # So instead we used sorted() normally, but for the key provide a tuple of
    # the negative value and the key.
    return [key for key, value in sorted(counts.items(), key=_items_sort_key)]


# DO NOT MODIFY
def _items_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    # This is used by items_descending_count, but you should never call it directly.
    return -item[1], item[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MostFrequentTagTagger(Tagger):
    def __init__(self) -> None:
        # Add an attribute to store the most frequent tag
        self.most_frequent_tag = None

    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        tag_counter = Counter()
        for sentence in sentences:
            for tagged_token in sentence:
                tag_counter[tagged_token.tag] += 1
        self.most_frequent_tag = most_frequent_item(tag_counter)

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        tags = []
        for i in range(len(sentence)):
            tags.append(self.most_frequent_tag)
        return tags


class UnigramTagger(Tagger):
    def __init__(self) -> None:
        # Add data structures that you need here
        self.most_frequent_tag = None
        self.mapping = {}

    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        tag_counter = Counter()
        mapping_types = {}
        for sentence in sentences:
            for tagged_token in sentence:
                tag_counter[tagged_token.tag] += 1
                if tagged_token.text not in mapping_types:
                    mapping_types[tagged_token.text] = Counter()
                mapping_types[tagged_token.text][tagged_token.tag] += 1
        self.most_frequent_tag = most_frequent_item(tag_counter)
        for key, counts in mapping_types.items():
            self.mapping[key] = most_frequent_item(counts)

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        tags = []
        for token in sentence:
            tags.append(self.mapping.get(token, self.most_frequent_tag))
        return tags

class SentenceCounter:
    def __init__(self, k: float) -> None:
        self.k = k
        # Add data structures that you need here
        self.tag_counter = Counter()
        self.token_counts_by_tag = {} # a dict of tags, each has the counts of all tokens it tagged
        self.tag_counts_by_tag = {}
        self.sorted_tags = []

    def count_sentences(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        for sentence in sentences:
            # Fill in this loop
            context = "<START>"
            for tagged_token in sentence:
                self.tag_counter[tagged_token.tag] += 1
                if tagged_token.tag not in self.token_counts_by_tag:
                    self.token_counts_by_tag[tagged_token.tag] = Counter()
                self.token_counts_by_tag[tagged_token.tag][tagged_token.text] += 1

                if context not in self.tag_counts_by_tag:
                    self.tag_counts_by_tag[context] = Counter()
                self.tag_counts_by_tag[context][tagged_token.tag] += 1
                context = tagged_token.tag # this tag becomes the context for the next tag
        self.sorted_tags = items_descending_value(self.tag_counter)

    def unique_tags(self) -> list[str]:
        return self.sorted_tags # should be all unique tags appeared more than 0

    def emission_prob(self, tag: str, word: str) -> float:
        k = self.k
        if tag not in self.token_counts_by_tag:
            return 0.0
        C = self.token_counts_by_tag[tag].get(word, 0)
        V = len(self.token_counts_by_tag[tag].keys())
        N = self.tag_counter[tag]
        try:
            return (C + k) / (N + k * V)
        except ZeroDivisionError:
            print(f"Division by zero when calculating emission "
                  f"probability for {word} given {tag}!")
            return 0.0

    def transition_prob(self, tag1: str, tag2: str) -> float:
        if tag1 not in self.tag_counts_by_tag:
            return 0.0
        numerator = self.tag_counts_by_tag[tag1].get(tag2, 0)
        denominator = self.tag_counts_by_tag[tag1].total()
        try:
            return numerator / denominator
        except ZeroDivisionError:
            print(f"The context {tag1} is not seen in training!")
            return 0.0

    def initial_prob(self, tag: str) -> float:
        numerator = self.tag_counts_by_tag["<START>"].get(tag, 0)
        denominator = self.tag_counts_by_tag["<START>"].total()
        return numerator / denominator # zero division ruled out in hw assumption

class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods and do not override __init__ or train in the subclasses.

    def __init__(self, k: float) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[TaggedToken]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        """Return the probability for a sequence of tags given tokens."""
        prob = safe_log(self.counter.initial_prob(tags[0]) *
                        self.counter.emission_prob(tags[0], sentence[0]))
        for i in range(len(sentence)-1):
            prob += safe_log(self.counter.transition_prob(tags[i], tags[i+1])
                             * self.counter.emission_prob(tags[i+1], sentence[i+1]))
        return prob

class GreedyBigramTagger(BigramTagger):
    # DO NOT IMPLEMENT AN __init__ METHOD

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        best_tags = [self.first_best_tag(sentence[0])]
        for i in range(len(sentence)-1):
            context = best_tags[i]
            tag_scores = {}
            for tag in self.counter.unique_tags():
                tag_scores[tag] = (safe_log(self.counter.emission_prob(tag, sentence[i+1]))
                                      + safe_log(self.counter.transition_prob(context, tag)))
            best_tags.append(max_item(tag_scores)[0])
        return best_tags

    def first_best_tag(self, token: str) -> str:
        tag_scores = {}
        for tag in self.counter.unique_tags():
                tag_scores[tag] = (safe_log(self.counter.initial_prob(tag))
                                      + safe_log(self.counter.emission_prob(tag, token)))
        return max_item(tag_scores)[0]


class ViterbiBigramTagger(BigramTagger):
    # DO NOT IMPLEMENT AN __init__ METHOD

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        viterbi = []
        back_pointers = []
        best_path = []

        tag_scores = {}
        pointer = {}
        for tag in self.counter.unique_tags():
            tag_scores[tag] = safe_log(self.counter.initial_prob(tag)) + safe_log(self.counter.emission_prob(tag, sentence[0]))
            pointer[tag] = 0
        viterbi.append(tag_scores)
        back_pointers.append(pointer)

        for i in range(len(sentence)-1):
            tag_scores = {}
            pointer = {}
            for tag in self.counter.unique_tags():
                find_tag = {}
                for tag1 in self.counter.unique_tags():
                    find_tag[tag1] = viterbi[i][tag1] + safe_log(self.counter.transition_prob(tag1, tag)) + safe_log(self.counter.emission_prob(tag, sentence[i+1]))
                tag_scores[tag] = max_item(find_tag)[1]
                pointer[tag] = max_item(find_tag)[0]

            viterbi.append(tag_scores)
            back_pointers.append(pointer)

        best_path.append(max_item(viterbi[len(sentence)-1])[0])

        for i in range(len(sentence)-1):
            # best_path[i]: starting from the best final state
            # len(back_pointers)-1-i: if 5 states, then index is 4,3,2,1
            # back_pointers[len(back_pointers)-1-i]: descending, {4:3; 3:2; 2:1; 1:0}
            best_path.append(back_pointers[len(back_pointers)-1-i][best_path[i]])
        best_path.reverse()
        return best_path