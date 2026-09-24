# Zepto Support Assistant

A policy-based customer support assistant built using RAG concepts, MiniLM embeddings, ChromaDB, LangGraph, JSON Schema, and FastAPI.

## Project Structure

```text
support_assistant/
├── data/
│   └── chroma/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── schema/
│   ├── request_schema.json
│   └── response_schema.json
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── graph.py
│   ├── generator_test.py
│   └── api.py
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md