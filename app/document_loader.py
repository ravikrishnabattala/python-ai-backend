from pathlib import Path

from app.pdf_loader import load_pdf


def load_documents(data_dir: str) -> str:
    data_path = Path(data_dir)

    all_text = []

    for file_path in data_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".pdf":
            text = load_pdf(str(file_path))

        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text(
                encoding="utf-8"
            )

        else:
            continue

        all_text.append(
            f"\n--- {file_path.name} ---\n{text}"
        )

    return "\n".join(all_text)