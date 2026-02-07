import random
from abc import abstractmethod
from math import log, prod
from pathlib import Path
from typing import Generator, Generic, Iterable, Sequence, TypeVar, Union
from collections import Counter, defaultdict
############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"
# DO NOT MODIFY
POS_INF = float("inf")
NEG_INF = float("-inf")
# DO NOT MODIFY
T = TypeVar("T")


# DO NOT MODIFY
def load_tokenized_file(path: Union[Path, str]) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            assert tokens, "Empty line in input"
            yield tuple(tokens)


# DO NOT MODIFY
def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order,
    # sort items before sampling. This is very slow and
    # should be avoided in general, but we do it in order
    # to get predictable results.
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weights in the values
    return random.choices(keys, weights=vals)[0]


# DO NOT MODIFY
class ProbabilityDistribution(Generic[T]):
    """A generic probability distribution."""

    # DO NOT ADD AN __INIT__ METHOD HERE

    # DO NOT MODIFY
    # You will implement this in subclasses
    @abstractmethod
    def prob(self, item: T) -> float:
        """Return a probability for the specified item."""
        raise NotImplementedError


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.

# Functions copied from hw1.py
def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    bigram_tuples = [('<START>',sentence[0])]
    for i in range(len(sentence)-1):
        bigram_tuples.append((sentence[i],sentence[i+1]))
    bigram_tuples.append((sentence[len(sentence)-1],'<END>'))
    return bigram_tuples

def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    trigram_tuples = [('<START>','<START>',sentence[0])]
    if len(sentence)<2:
        trigram_tuples.append(('<START>', sentence[0], '<END>'))
        return trigram_tuples
    trigram_tuples.append(('<START>',sentence[0],sentence[1]))
    for i in range(len(sentence)-2):
        trigram_tuples.append((sentence[i],sentence[i+1],sentence[i+2]))
    trigram_tuples.append((sentence[len(sentence)-2],sentence[len(sentence)-1],'<END>'))
    return trigram_tuples

def unigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, int]:
    c = Counter()
    for sentence in sentences:
        c.update(sentence)
    uni_counts = dict(c)
    return uni_counts


def bigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, dict[str, int]]:
    d = defaultdict(Counter)
    for sentence in sentences:
        bigram_tuples = bigrams(sentence)
        for token1, token2 in bigram_tuples:
            d[token1][token2] += 1
    bi_counts = {}
    for key, val in d.items():
        bi_counts[key] = dict(val)
    return bi_counts


def trigram_counts(
        sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, int]]:
    d = defaultdict(Counter)
    for sentence in sentences:
        trigram_tuples = trigrams(sentence)
        for token1, token2, token3 in trigram_tuples:
            d[token1,token2][token3] += 1
    tri_counts = {}
    for key, val in d.items():
        tri_counts[key] = dict(val)
    return tri_counts

def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    context = sample(probs[START_TOKEN])
    sampled = []
    while context != END_TOKEN:
        sampled.append(context)
        context = sample(probs[context])
    return sampled



def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    context = (START_TOKEN, sample(probs[START_TOKEN, START_TOKEN]))
    sampled = []
    while context[1] != END_TOKEN:
        sampled.append(context[1])
        new_context = sample(probs[context])
        context = (context[1], new_context)
    return sampled

class UnigramMLE(ProbabilityDistribution[str]):
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.total = sum(counts.values())

    def prob(self, item: str) -> float:
        count = self.counts.get(item, 0)
        return count / self.total


class BigramMLE(ProbabilityDistribution[tuple[str, str]]):
    def __init__(self, counts: dict[str, dict[str, int]]) -> None:
        self.counts = counts
        self.total = {}
        for key in counts:
            self.total[key] = sum(counts[key].values())

    def prob(self, item: tuple[str, str]) -> float:
        if item[0] in self.counts:
            items = self.counts[item[0]].get(item[1], 0)
            return items / self.total[item[0]]
        return 0.0

class TrigramMLE(ProbabilityDistribution[tuple[str, str, str]]):
    def __init__(self, counts: dict[tuple[str, str], dict[str, int]]) -> None:
        self.counts = counts
        self.total = {}
        for key in counts:
            self.total[key] = sum(counts[key].values())

    def prob(self, item: tuple[str, str, str]) -> float:
        if (item[0],item[1]) in self.counts:
            items = self.counts[(item[0], item[1])].get(item[2], 0)
            return items / self.total[item[0], item[1]]
        return 0.0


class UnigramLidstoneSmoothed(ProbabilityDistribution[str]):
    def __init__(self, counts: dict[str, int], k: float) -> None:
        self.counts = counts
        self.k = k
        self.total = sum(counts.values())

    def prob(self, item: str) -> float:
        count = self.counts.get(item)
        if count is None:
            return 0.0
        return (count + self.k) / (self.total + len(self.counts)*self.k)


class BigramInterpolation(ProbabilityDistribution[tuple[str, str]]):
    def __init__(
        self,
        uni_probs: ProbabilityDistribution[str],
        bi_probs: ProbabilityDistribution[tuple[str, str]],
        l_1: float,
        l_2: float,
    ) -> None:
        self.uni_probs = uni_probs
        self.bi_probs = bi_probs
        self.l_1 = l_1
        self.l_2 = l_2

    def prob(self, item: tuple[str, str]) -> float:
        uni_prob = self.uni_probs.prob(item[1])
        bi_prob = self.bi_probs.prob(item)
        return uni_prob * self.l_1 + bi_prob * self.l_2


def unigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[str]
) -> float:
    uni_seq_prob = 0
    for seq in sequence:
        if probs.prob(seq) == 0:
            uni_seq_prob = NEG_INF
            break
        uni_seq_prob += log(probs.prob(seq))
    return uni_seq_prob


def bigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str]]
) -> float:
    bi_seq_prob = 0
    sequence = bigrams(sequence)
    for seq in sequence:
        if probs.prob(seq) == 0:
            bi_seq_prob = NEG_INF
            break
        bi_seq_prob += log(probs.prob(seq))
    return bi_seq_prob


def trigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str, str]]
) -> float:
    tri_seq_prob = 0
    sequence = trigrams(sequence)
    for seq in sequence:
        if probs.prob(seq) == 0:
            tri_seq_prob = NEG_INF
            break
        tri_seq_prob += log(probs.prob(seq))
    return tri_seq_prob


def compute_perplexity(
    probabilities: Iterable[float], n: int
) -> float:
    if n == 0:
        return POS_INF
    return (prod(probabilities))**(1/n)

def unigram_perplexity(
    sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[str]
) -> float:
    count_tokens = 0
    probabilities = []
    for sentence in sentences:
        prob = 1
        for token in sentence:
            count_tokens += 1
            if probs.prob(token) == 0:
                perplexity = POS_INF
                return perplexity
            prob/=(probs.prob(token))
        probabilities.append(prob)
    return compute_perplexity(probabilities, count_tokens)


def bigram_perplexity(
    sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[tuple[str, str]]
) -> float:
    count_tokens = 0
    probabilities = []
    for sentence in sentences:
        sentence = bigrams(sentence)
        prob = 1
        for token in sentence:
            count_tokens += 1
            if probs.prob(token) == 0:
                perplexity = POS_INF
                return perplexity
            prob /= (probs.prob(token))
        probabilities.append(prob)
    return compute_perplexity(probabilities, count_tokens)


def trigram_perplexity(
    sentences: Iterable[Sequence[str]],
    probs: ProbabilityDistribution[tuple[str, str, str]],
) -> float:
    count_tokens = 0
    probabilities = []
    for sentence in sentences:
        sentence = trigrams(sentence)
        prob = 1
        for token in sentence:
            count_tokens += 1
            if probs.prob(token) == 0:
                perplexity = POS_INF
                return perplexity
            prob /= (probs.prob(token))
        probabilities.append(prob)
    return compute_perplexity(probabilities, count_tokens)
