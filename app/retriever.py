import math

from app.embeddings import tokenize


def cosine_similarity(query_vector, chunk_vector):
    common_words = set(query_vector) & set(chunk_vector)

    dot_product = sum(
        query_vector[word] * chunk_vector[word]
        for word in common_words
    )

    query_magnitude = math.sqrt(
        sum(value * value for value in query_vector.values())
    )

    chunk_magnitude = math.sqrt(
        sum(value * value for value in chunk_vector.values())
    )

    if query_magnitude == 0 or chunk_magnitude == 0:
        return 0.0

    return dot_product / (query_magnitude * chunk_magnitude)


def retrieve_chunks(
    query: str,
    chunks: list[str],
    embeddings,
    top_k: int = 1
):
    query_tokens = tokenize(query)

    query_vector = {}

    for token in query_tokens:
        query_vector[token] = query_vector.get(token, 0) + 1

    results = []

    for index, chunk_vector in enumerate(embeddings):
        score = cosine_similarity(
            query_vector,
            chunk_vector
        )

        results.append({
            "chunk": chunks[index],
            "score": score
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]