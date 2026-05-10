# text_stats.py

from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def quote_per_1000_words(text: str) -> float:
    """Count quote characters and normalize per 1000 words."""
    quote_chars = ['"', "„", "“", "‚", "‘", "’", "«", "»"]
    n_quotes = sum(text.count(q) for q in quote_chars)
    n_words = max(len(text.split()), 1)
    return n_quotes / n_words * 1000


def sentence_quote_ratio(text: str, language: str = "german") -> float:
    """Share of sentences that contain at least one quote character."""
    sentences = sent_tokenize(text, language=language)
    quoted = sum(('"' in s) or ("„" in s) for s in sentences)
    return quoted / max(len(sentences), 1)


def sentence_lengths_sampled(
    text: str, max_sentences: int = 50_000, language: str = "german"
) -> list[int]:
    """Sample up to max_sentences and return sentence lengths in word tokens."""
    sentences = sent_tokenize(text, language=language)
    if len(sentences) > max_sentences:
        sentences = random.sample(sentences, max_sentences)
    return [len(word_tokenize(s, language=language)) for s in sentences]


def word_lengths_sampled(text: str, max_words: int = 200_000) -> list[int]:
    """Sample up to max_words and return word lengths in characters."""
    words = [w for w in text.split() if w.isalpha()]
    if len(words) > max_words:
        words = random.sample(words, max_words)
    return [len(w) for w in words]


def stopword_ratio(text: str, language: str = "german") -> float:
    """Share of words that are stopwords (function words)."""
    stops = set(stopwords.words(language))
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    return sum(w in stops for w in words) / len(words)


def type_token_ratio(words) -> float:
    """
    Type–Token Ratio (TTR):
    unique_words / total_words. Higher => more lexical variety.

    Note: TTR decreases with text length, so it’s best used for comparisons
    (e.g., comparing corpora rather than interpreting absolute values).
    """
    return len(set(words)) / max(len(words), 1)


def gini_coefficient(counter: Counter) -> float:
    """
    Gini coefficient over word frequencies:
    measures how unequal the frequency distribution is.

    - High Gini: a few words dominate (high repetition / concentration)
    - Low Gini: frequencies are more evenly spread (less concentration)
    """
    freqs = np.array(list(counter.values()))
    freqs = np.sort(freqs)
    n = len(freqs)
    cum = np.cumsum(freqs)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def top_words_normalized(text: str, n: int = 20):
    """
    Top-N single words (unigrams), normalized to 'per 10k words' for comparability.
    """
    words = [w.lower() for w in text.split() if w.isalpha()]
    total = max(len(words), 1)
    counts = Counter(words).most_common(n)
    return [(w, c / total * 10_000) for w, c in counts]


def top_bigrams_normalized(text: str, n: int = 20):
    """
    Top-N word bigrams (consecutive word pairs), normalized to 'per 100k bigrams'
    so corpora of different sizes can be compared.
    """
    words = [w.lower() for w in text.split() if w.isalpha()]
    bigrams = list(zip(words[:-1], words[1:]))
    total = max(len(bigrams), 1)
    counts = Counter(bigrams).most_common(n)
    return [((w1, w2), c / total * 100_000) for (w1, w2), c in counts]


def top_bigrams_no_stopwords(text: str, n: int = 20, language: str = "german"):
    """
    Top-N bigrams after removing stopwords.
    This tends to surface content/semantic patterns instead of syntactic patterns.
    """
    stops = set(stopwords.words(language))
    words = [w.lower() for w in text.split() if w.isalpha() and w.lower() not in stops]
    bigrams = list(zip(words[:-1], words[1:]))
    total = max(len(bigrams), 1)
    counts = Counter(bigrams).most_common(n)
    return [((w1, w2), c / total * 100_000) for (w1, w2), c in counts]


def strip_wikipedia_sections(text: str) -> str:
    """
    Remove common non-prose sections from Wikipedia pages (e.g. Literature, References).
    This reduces artifacts like 'ISBN' showing up in content bigrams.
    """
    patterns = [
        r"\n==\s*literatur\s*==.*$",
        r"\n==\s*einzelnachweise\s*==.*$",
        r"\n==\s*weblinks\s*==.*$",
        r"\n==\s*web links\s*==.*$",
        r"\n==\s*quellen\s*==.*$",
        r"\n==\s*referenzen\s*==.*$",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out
