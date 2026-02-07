from collections import Counter
from typing import Generator, Iterable


def sorted_chars(s: str) -> list[str]:
    chars = []
    for char in s:
        if char not in chars:
            chars.append(char)
    chars = sorted(chars)
    return chars


def gen_sentences(path: str) -> Generator[list[str], None, None]:
    with open(path, encoding="utf8") as file:
        for line in file:
            if line.strip() == "": continue
            line = line.rstrip('\n')
            sentences = line.split(" ")
            yield sentences


def n_most_frequent_tokens(sentences: Iterable[list[str]], n: int) -> list[str]:
    if n < 0:
        raise ValueError('n cannot be negative')
    tokens = []
    for sentence in sentences:
        for token in sentence:
            tokens.append(token)
    c = Counter(tokens).most_common(n)
    nmf_tokens = []
    for t in c:
        nmf_tokens.append(t[0])
    return nmf_tokens


def case_sarcastically(text: str) -> str:
    cased = ''
    state = 1
    for t in text:
        if t.upper() == t.lower():
            cased += t
        elif state == 0:
            cased += t.upper()
            state = 1
        else:
            cased += t.lower()
            state = 0
    return cased


