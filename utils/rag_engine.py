import os
from typing import List, Dict, Any, Callable
from collections import defaultdict
import numpy as np

# --- Simple Embedding (TF-IDF style, no external dependencies) ---
class SimpleEmbedding:
    def __init__(self):
        self.vocabulary = set()
        self.idf = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in text.split() if w.isalnum()]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        tf = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = len(tokens)
        return {k: v / total for k, v in tf.items()}

    def fit(self, documents: List[str]):
        doc_count = defaultdict(int)
        N = len(documents)
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_count[t] += 1
        self.vocabulary = set(doc_count.keys())
        self.idf = {t: np.log((N + 1) / (doc_count[t] + 1)) + 1 for t in self.vocabulary}
        self.fitted = True

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Embedding model not fitted.")
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = self._compute_tf(tokens)
            vec = np.array([tf.get(t, 0) * self.idf.get(t, 0) for t in self.vocabulary])
            vectors.append(vec)
        return np.array(vectors)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# --- Document Chunker ---
class DocumentChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta['text'] = chunk_text
            chunks.append(chunk_meta)
            i += self.chunk_size - self.overlap
        return chunks

# --- Vector Store ---
class VectorStore:
    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.documents = []
        self.metadata = []
        self.embeddings = None

    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        self.documents.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
        self.embedding_model.fit(self.documents)
        self.embeddings = self.embedding_model.transform(self.documents)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        query_vec = self.embedding_model.transform([query])[0]
        sims = [self.embedding_model.cosine_similarity(query_vec, doc_vec) for doc_vec in self.embeddings]
        top_k_idx = np.argsort(sims)[::-1][:k]
        return [dict(self.metadata[i], text=self.documents[i], score=float(sims[i])) for i in top_k_idx]

# --- RAG Engine ---
class SimpleRAG:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunker = DocumentChunker(chunk_size, overlap)
        self.embedding_model = SimpleEmbedding()
        self.vector_store = VectorStore(self.embedding_model)
        self.workflow_steps = []

    def log_step(self, step: str, status: str = "info", details: str = ""):
        from datetime import datetime
        self.workflow_steps.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "status": status,
            "details": details
        })

    def build_knowledge_base(self, documents: List[Dict], progress_callback: Callable = None):
        all_chunks = []
        all_metadata = []
        for i, doc in enumerate(documents):
            text = doc.get('text', '')
            meta = {k: v for k, v in doc.items() if k != 'text'}
            chunks = self.chunker.chunk_text(text, meta)
            all_chunks.extend([c['text'] for c in chunks])
            all_metadata.extend(chunks)
            if progress_callback:
                progress_callback((i + 1) / len(documents), f"Chunked document {i + 1}/{len(documents)}")
        self.vector_store.add_documents(all_chunks, all_metadata)
        self.log_step("Knowledge base built", status="success", details=f"{len(all_chunks)} chunks indexed.")

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        return self.vector_store.similarity_search(query, k=k)

    def generate_response(self, query: str, context: List[Dict], ai_model: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 1500) -> str:
        # Placeholder: In production, call LLM API with context and query
        context_text = '\n'.join([c['text'] for c in context])
        return f"[CONTEXT]\n{context_text}\n[QUERY]\n{query}\n[RESPONSE]\n(This is a placeholder. Integrate with LLM for real response.)"

    def query(self, question: str, k: int = 5) -> Dict:
        self.log_step(f"User query: {question}", status="processing")
        context = self.retrieve_context(question, k=k)
        response = self.generate_response(question, context)
        self.log_step(f"Response generated", status="success")
        return {"response": response, "context": context, "workflow": self.workflow_steps}
