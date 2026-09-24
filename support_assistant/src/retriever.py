from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def search_documents(query, top_k=3):
    

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name="zepto_support"
    )

    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


if __name__ == "__main__":
    query = "What is the weather today?"
    results = search_documents(query)

    print(f"Query: {query}")
    print("\nRetrieved documents:\n")

    for i, document in enumerate(results["documents"][0], start=1):
        distance = results["distances"][0][i - 1]
        source = results["metadatas"][0][i - 1]["source"]

        print(f"--- Result {i} ---")
        print(f"Source: {source}")
        print(f"Distance: {distance}")
        print(document)
        print()