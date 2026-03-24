from rank_bm25 import BM25Okapi

# ==================== BM25 Retrieval ====================
class BM25Retriever:
    """BM25-based document retriever."""

    def __init__(self, corpus_docs):
        self.corpus_docs = corpus_docs
        self.titles = [doc[0] for doc in corpus_docs]
        self.texts = [doc[1] for doc in corpus_docs]
        tokenized_corpus = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text):
        return text.lower().split()

    def search(self, query, top_k=3):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append((self.titles[idx], self.texts[idx], float(scores[idx])))
        return results

    def lookup(self, title):
        title_lower = title.lower().strip()
        for t, text in self.corpus_docs:
            if t.lower().strip() == title_lower:
                return text
        return None
