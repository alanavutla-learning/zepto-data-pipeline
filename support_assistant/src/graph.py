import json
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from jsonschema import validate

from .retriever import search_documents

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schema"
    / "response_schema.json"
)

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    RESPONSE_SCHEMA = json.load(file)

class SupportState(TypedDict):
    question: str
    retrieved_documents: list
    response: dict
def retrieve_node(state: SupportState):
    results = search_documents(
        state["question"],
        top_k=3,
    )

    documents = []

    for i, document in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]

        documents.append(
            {
                "source": source,
                "content": document,
                "distance": results["distances"][0][i],
            }
        )

    return {
        "retrieved_documents": documents
    }


def answer_node(state: SupportState):
    documents = state["retrieved_documents"]

    if not documents:
        response = {
            "answer": "I don't have enough information in the available policy documents.",
            "source": "",
        }

        validate(instance=response, schema=RESPONSE_SCHEMA)

        return {"response": response}

    answer = documents[0]["content"]
    source = documents[0]["source"]

    response = {
        "answer": answer,
        "source": source,
    }

    validate(instance=response, schema=RESPONSE_SCHEMA)

    return {"response": response}

def route_after_retrieval(state: SupportState):
    documents = state["retrieved_documents"]

    if not documents:
        return "fallback"

    distance = documents[0]["distance"]

    if distance <= 1.4:
        return "answer"

    return "fallback"

def fallback_node(state: SupportState):
    response = {
        "answer": (
            "I don't have enough information in the available "
            "policy documents to answer this question."
        ),
        "source": "",
    }

    validate(instance=response, schema=RESPONSE_SCHEMA)

    return {"response": response}
def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()

if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(
        {
            "question": "What happens if my order arrives damaged?",
            "retrieved_documents": [],
            "response": {},
        }
    )

    print("Question:")
    print(result["question"])

    print("\nResponse:")
    print(result["response"])