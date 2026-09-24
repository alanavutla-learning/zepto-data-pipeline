from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from loader import load_documents
from splitter import split_documents


CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"


def create_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="zepto_support"
    )

    texts = [chunk.page_content for chunk in chunks]

    vectors = embeddings.embed_documents(texts)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {"source": chunk.metadata["source"]}
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )

    return collection


if __name__ == "__main__":
    collection = create_vector_store()

    print("Vector store created successfully.")
    print(f"Stored chunks: {collection.count()}")