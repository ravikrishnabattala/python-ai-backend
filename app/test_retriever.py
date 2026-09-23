from pathlib import Path

from app.chunker import chunk_text
from app.embeddings import create_embeddings
from app.pdf_loader import load_pdf
from app.retriever import retrieve_chunks


pdf_path = Path(__file__).parent.parent / "data" / "resume.pdf"

resume_text = load_pdf(str(pdf_path))

chunks = chunk_text(resume_text)

embeddings = create_embeddings(chunks)

query = "What automation tools does Ravi know?"

results = retrieve_chunks(
    query=query,
    chunks=chunks,
    embeddings=embeddings,
    top_k=1
)

print(f"\nQuery: {query}")

for index, result in enumerate(results):
    print(f"\n--- Result {index + 1} ---")
    print(f"Similarity: {result['score']:.4f}")
    print(f"Chunk:\n{result['chunk']}")