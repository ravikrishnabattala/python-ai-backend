from pathlib import Path

from pypdf import PdfReader

from app.chunker import chunk_text


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "data" / "resume.pdf"

    resume_text = load_pdf(str(pdf_path))

    chunks = chunk_text(resume_text)

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- Chunk {index + 1} ---")
        print(chunk)