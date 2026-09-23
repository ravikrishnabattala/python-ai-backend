import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9+#.-]+\b", text.lower())


def create_embeddings(chunks: list[str]):
    return [
        Counter(tokenize(chunk))
        for chunk in chunks
    ]