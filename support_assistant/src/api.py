import json
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException
from jsonschema import validate

from .graph import build_graph


app = FastAPI(
    title="Zepto Support Assistant"
)

graph = build_graph()

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schema"
    / "request_schema.json"
)

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    REQUEST_SCHEMA = json.load(file)


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant API is running."
    }


@app.post("/ask")
async def ask_question(data: dict = Body(...)):
    try:
        validate(
            instance=data,
            schema=REQUEST_SCHEMA,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    result = graph.invoke(
        {
            "question": data["question"],
            "retrieved_documents": [],
            "response": {},
        }
    )

    return result["response"]