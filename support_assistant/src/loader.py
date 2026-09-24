from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def load_documents():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "content": text,
            }
        )

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded documents: {len(documents)}")

    for document in documents:
        print(
            f"\n--- {document['source']} ---\n"
            f"{document['content'][:200]}"
        )