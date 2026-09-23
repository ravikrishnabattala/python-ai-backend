from pathlib import Path

from app.chunker import chunk_text
from app.embeddings import create_embeddings
from app.pdf_loader import load_pdf


pdf_path = Path(__file__).parent.parent / "data" / "resume.pdf"

resume_text = load_pdf(str(pdf_path))

chunks = chunk_text(resume_text)

embeddings = create_embeddings(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

for index, embedding in enumerate(embeddings):
    print(f"Chunk {index + 1} vector size:", len(embedding))