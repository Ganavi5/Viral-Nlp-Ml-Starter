from typing import Iterable
from sklearn.feature_extraction.text import CountVectorizer
from .kmer import make_kmers

class KmerVectorizer:
    def __init__(self, k: int = 6, **kwargs):
        # token_pattern ensures we don't ignore short 'words'
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\\b\\w+\\b", **kwargs)
        self.k = k

    def _kmers_as_strings(self, sequences: Iterable[str]):
        return [" ".join(make_kmers(seq, self.k)) for seq in sequences]

    def fit(self, sequences: Iterable[str]):
        corpus = self._kmers_as_strings(sequences)
        self.vectorizer.fit(corpus)
        return self

    def transform(self, sequences: Iterable[str]):
        corpus = self._kmers_as_strings(sequences)
        return self.vectorizer.transform(corpus)

    def fit_transform(self, sequences: Iterable[str]):
        corpus = self._kmers_as_strings(sequences)
        return self.vectorizer.fit_transform(corpus)
