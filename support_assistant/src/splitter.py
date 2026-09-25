from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import load_documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = []

    for document in documents:
        document_chunks = splitter.create_documents(
            texts=[document["content"]],
            metadatas=[{"source": document["source"]}],
        )

        chunks.extend(document_chunks)

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(chunk.page_content[:200])