import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_chunks(query: str, chunks: list[str], embeddings, top_k: int = 1):
    query_embedding = model.encode([query])[0]

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(similarities[index])
        })

    return results