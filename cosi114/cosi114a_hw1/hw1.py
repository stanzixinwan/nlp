from collections import Counter, defaultdict
from typing import Iterable, Sequence, TypeVar

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"


def counts_to_probs(counts: dict[T, int]) -> dict[T, float]:
    probs = {}
    num_words = sum(counts.values())
    for key,val in counts.items():
        probs[key] = float(val) / num_words
    return probs


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


def unigram_probs(counts: dict[str, int]) -> dict[str, float]:
    return counts_to_probs(counts)


def bigram_probs(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    b = {}
    for key, val in counts.items():
        b[key] = counts_to_probs(val)
    return b


def trigram_probs(
    counts: dict[tuple[str, str], dict[str, int]],
) -> dict[tuple[str, str], dict[str, float]]:
    c = {}
    for key, val in counts.items():
        c[key] = counts_to_probs(val)
    return c

