import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .intent import classify_intent
from .retriever import search_documents
from .schemas import SupportResponse
from .generator import generate_answer


class SupportState(TypedDict, total=False):
    question: str
    intent: str
    retrieved_documents: list
    response: dict


def classify_intent_node(state: SupportState):
    intent = classify_intent(state["question"])
    return {"intent": intent}


def retrieve_and_answer(state: SupportState):
    results = search_documents(state["question"], top_k=3)

    documents = []

    for i, document in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]

        documents.append(
            {
                "source": source,
                "content": document,
                "distance": distance,
            }
        )

    if not documents:
        return {
            "retrieved_documents": [],
           "response": SupportResponse(
                            answer="Based on the retrieved context: No relevant policy information was found.",
                            sources=[],
                            confidence=0.0,
                            ).model_dump(),       }

    top_document = documents[0]

    answer = generate_answer(
    state["question"],
    top_document["content"],
    )

    return {
        "retrieved_documents": documents,
        "response": {
            "answer": answer,
            "sources": [top_document["source"]],
            "confidence": 1.0,
        },
    }


def direct_answer(state: SupportState):
    return {
        "response": SupportResponse(
            answer="I can help with Zepto support questions, but this question is outside the available policy topics.",
            sources=[],
            confidence=0.0,
        ).model_dump(),
    }


def route_by_intent(state: SupportState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()