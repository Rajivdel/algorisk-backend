# utils/rag_engine.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import uuid

class SimpleEmbedding:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

class DocumentChunker:
    def __init__(self, chunk_size=300):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.documents = []
        self.ids = []
        self.vectorizer = SimpleEmbedding()

    def index(self, docs: List[Dict[str, str]]):
        all_chunks = []
        self.documents = []
        self.ids = []
        chunker = DocumentChunker()
        for doc in docs:
            chunks = chunker.chunk(doc["content"])
            for chunk in chunks:
                all_chunks.append(chunk)
                self.documents.append({"content": chunk, "meta": doc})
                self.ids.append(str(uuid.uuid4()))
        self.embeddings = self.vectorizer.fit_transform(all_chunks)
        return len(all_chunks)

    def query(self, query_text: str, top_k: int = 3):
        if not self.embeddings:
            return []
        query_embedding = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] | {"similarity": similarities[i]} for i in top_indices]

class SimpleRAG:
    def __init__(self):
        self.vs = VectorStore()

    def build_knowledge_base(self, docs: List[Dict[str, str]]) -> int:
        return self.vs.index(docs)

    def query(self, question: str) -> Dict:
        retrieved_docs = self.vs.query(question, top_k=3)
        response = "\n---\n".join([f"{doc['content']}" for doc in retrieved_docs])
        return {
            "response": f"Based on retrieved documents, here is the answer:\n{response}",
            "workflow": [doc["meta"] for doc in retrieved_docs]
        }
