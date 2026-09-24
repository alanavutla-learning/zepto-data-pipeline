from langchain_huggingface import HuggingFaceEmbeddings

from splitter import split_documents
from loader import load_documents


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector = embeddings.embed_query(
        "Can I cancel my Zepto order?"
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")