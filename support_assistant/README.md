# Zepto Support Assistant â€” Module 3

A policy-based customer support assistant built using Retrieval-Augmented Generation (RAG) concepts, MiniLM embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

The assistant uses an 8-document Zepto policy corpus. Policy questions are classified, relevant policy chunks are retrieved from ChromaDB, and a mock response is generated from the top retrieved chunk. Unrelated questions are routed to a direct canned response.

## 1. Module Purpose

The Support Assistant provides a small RAG-based customer support pipeline for answering questions from the supplied Zepto policy documents.

The module demonstrates:

* Document ingestion
* Text splitting
* Sentence-transformer embeddings
* ChromaDB vector storage
* Semantic retrieval
* Keyword-based intent classification
* LangGraph conditional routing
* Mock generation
* Pydantic response validation
* FastAPI API serving
* Docker configuration
* Optional real-LLM extension with structured prompting and retry logic

The default configuration uses:

```text
MOCK_LLM=1
```

No external LLM API call is made in the default mode.

---

## 2. Corpus

The corpus contains eight policy documents:

```text
docs/
â”œâ”€â”€ doc_01.txt
â”œâ”€â”€ doc_02.txt
â”œâ”€â”€ doc_03.txt
â”œâ”€â”€ doc_04.txt
â”œâ”€â”€ doc_05.txt
â”œâ”€â”€ doc_06.txt
â”œâ”€â”€ doc_07.txt
â””â”€â”€ doc_08.txt
```

The documents are loaded by `src/loader.py`.

They are split into chunks by `src/splitter.py`, embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and stored in the ChromaDB collection:

```text
zepto_support
```

The verified local ChromaDB count is 8 chunks.

---

## 3. Architecture

The RAG pipeline is:

```text
INGESTION
    â†“
EMBEDDING
    â†“
RETRIEVAL
    â†“
GENERATION
```

### Stage 1 â€” Ingestion

**Files/components:**

```text
src/loader.py
src/splitter.py
docs/doc_01.txt ... docs/doc_08.txt
```

`loader.py` reads all eight text documents.

`splitter.py` uses `RecursiveCharacterTextSplitter` to divide the documents into searchable chunks.

The vector store initialization is performed by:

```text
src/vector_store.py
```

---

### Stage 2 â€” Embedding

**Component:**

```text
src/vector_store.py
```

**Model:**

```text
sentence-transformers/all-MiniLM-L6-v2
```

The text chunks are converted into numerical embedding vectors.

The embeddings are stored in the ChromaDB collection:

```text
zepto_support
```

The persistent database location is:

```text
data/chroma/
```

The application automatically creates or updates the vector store during startup, allowing a fresh environment such as Docker to initialize the database from the corpus.

---

### Stage 3 â€” Retrieval

**File:**

```text
src/retriever.py
```

**LangGraph node:**

```text
retrieve_and_answer
```

For a policy question, the graph calls the retriever.

The query is converted into an embedding and compared against the ChromaDB collection.

The top three results are retrieved.

The highest-ranked retrieved chunk is used as the context for mock generation.

The source filename is returned in the `sources` field.

---

### Stage 4 â€” Generation

**File:**

```text
src/generator.py
```

**LangGraph nodes:**

```text
retrieve_and_answer
direct_answer
```

When:

```text
MOCK_LLM=1
```

the policy response is generated using the fixed template:

```text
Based on the retrieved context: <top retrieved chunk>
```

No network LLM call is made.

For unrelated questions, `direct_answer` returns the fixed canned response:

```text
I can help with Zepto support questions, but this question is outside the available policy topics.
```

---

## 4. Intent Classification and Routing

**File:**

```text
src/intent.py
```

The default `classify_intent` implementation uses keyword heuristics.

Policy keywords include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

For example:

```text
Can I cancel my order?
```

is classified as:

```text
policy_question
```

while:

```text
What is the weather today?
```

is classified as:

```text
general_question
```

No LLM call is required for this classification in default mock mode.

---

## 5. LangGraph

The graph contains three named nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The routing flow is:

```text
START
  â†“
classify_intent
  â†“
  â”œâ”€â”€ policy_question â”€â”€â†’ retrieve_and_answer â”€â”€â†’ END
  â”‚
  â””â”€â”€ general_question â”€â†’ direct_answer â”€â”€â”€â”€â”€â”€â”€â†’ END
```

### Policy example

Question:

```text
Can I cancel my order?
```

Route:

```text
classify_intent
    â†“
policy_question
    â†“
retrieve_and_answer
```

The retrieved source is:

```text
doc_05.txt
```

### General example

Question:

```text
What is the weather today?
```

Route:

```text
classify_intent
    â†“
general_question
    â†“
direct_answer
```

---

## 6. Structured Prompt

The structured prompt is defined in:

```text
src/prompt.py
```

It contains the main prompt components:

* Assistant role/instruction
* Retrieved context
* User question
* Output format
* Constraints

It also contains an explicit negative constraint:

```text
Do not invent, assume, or add policy information that is not present in the retrieved context.
```

A few-shot example is also included.

Example:

```text
Question: Can I cancel my order?

Context: Orders can be cancelled before the order status changes to 'Packed'.

Answer: Orders can be cancelled before they are packed.
```

The structured prompt is prepared for the optional real-LLM path.

---

## 7. MOCK_LLM Modes

### Default mode

```text
MOCK_LLM=1
```

This is the default.

The assistant does not make an external LLM call.

The retrieval stage still runs normally for policy questions.

The answer is constructed from the top retrieved chunk.

### Optional real-LLM mode

```text
MOCK_LLM=0
```

The code path in `src/generator.py` prepares the structured prompt for a real LLM provider.

The implementation contains retry-on-failure logic with two attempts.

A real provider/API must be configured before enabling this mode.

The default submission mode remains:

```text
MOCK_LLM=1
```

---

## 8. Response Schema

The API response is validated using Pydantic in:

```text
src/schemas.py
```

The response contains:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

The fields are:

| Field        | Type            | Description                      |
| ------------ | --------------- | -------------------------------- |
| `answer`     | string          | Generated or canned answer       |
| `sources`    | list of strings | Retrieved source document names  |
| `confidence` | float           | Confidence value between 0 and 1 |

Example policy response:

```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled...",
  "sources": ["doc_05.txt"],
  "confidence": 1.0
}
```

Example general response:

```json
{
  "answer": "I can help with Zepto support questions, but this question is outside the available policy topics.",
  "sources": [],
  "confidence": 0.0
}
```

---

## 9. FastAPI

**File:**

```text
src/api.py
```

The application exposes:

```text
GET /
POST /ask
```

Run locally with:

```powershell
uvicorn src.api:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

### Example 1 â€” Policy question

Request:

```json
{
  "question": "Can I cancel my order?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.",
  "sources": ["doc_05.txt"],
  "confidence": 1.0
}
```

### Example 2 â€” Unrelated question

Request:

```json
{
  "question": "What is the weather today?"
}
```

Response:

```json
{
  "answer": "I can help with Zepto support questions, but this question is outside the available policy topics.",
  "sources": [],
  "confidence": 0.0
}
```

Both example calls were verified locally with the default `MOCK_LLM=1` configuration.

---

## 10. Validation and Tests

The request schema is defined in:

```text
schema/request_schema.json
```

The response model is defined using Pydantic in:

```text
src/schemas.py
```

The API test suite is:

```text
tests/test_api.py
```

The current test result is:

```text
5 passed
```

The only displayed warning is a Starlette/AnyIO deprecation warning and does not cause test failure.

---

## 11. Docker

A Dockerfile is provided at:

```text
support_assistant/Dockerfile
```

It runs the FastAPI application using Uvicorn.

Build the image locally:

```powershell
docker build -t zepto-support-assistant .
```

Run the container:

```powershell
docker run -p 8000:8000 zepto-support-assistant
```

Then access:

```text
http://127.0.0.1:8000
```

and send requests to:

```text
POST /ask
```

The application initializes the ChromaDB vector store from the eight corpus documents when it starts, so the Docker image does not depend on a previously created local Chroma database.

Docker build/run was not executed in the development environment because Docker CLI was not installed there.

---

## 12. Project Structure

```text
support_assistant/
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”‚
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ doc_01.txt
â”‚   â”œâ”€â”€ doc_02.txt
â”‚   â”œâ”€â”€ doc_03.txt
â”‚   â”œâ”€â”€ doc_04.txt
â”‚   â”œâ”€â”€ doc_05.txt
â”‚   â”œâ”€â”€ doc_06.txt
â”‚   â”œâ”€â”€ doc_07.txt
â”‚   â””â”€â”€ doc_08.txt
â”‚
â”œâ”€â”€ schema/
â”‚   â””â”€â”€ request_schema.json
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ api.py
â”‚   â”œâ”€â”€ embedder.py
â”‚   â”œâ”€â”€ generator.py
â”‚   â”œâ”€â”€ graph.py
â”‚   â”œâ”€â”€ intent.py
â”‚   â”œâ”€â”€ loader.py
â”‚   â”œâ”€â”€ prompt.py
â”‚   â”œâ”€â”€ retriever.py
â”‚   â”œâ”€â”€ schemas.py
â”‚   â”œâ”€â”€ splitter.py
â”‚   â””â”€â”€ vector_store.py
â”‚
â””â”€â”€ tests/
    â””â”€â”€ test_api.py
```

The local ChromaDB directory is:

```text
data/chroma/
```

and is excluded from Git because it is generated locally.

---

## 13. Module 3 Acceptance Summary

The implementation demonstrates:

* Eight corpus documents loaded and embedded.
* ChromaDB retrieval from the `zepto_support` collection.
* Structured prompt with constraints, negative constraint, and few-shot example.
* Default keyword-based intent classification.
* Three-node LangGraph with conditional routing.
* Policy retrieval from the appropriate source document.
* Mock generation using the top retrieved chunk.
* Fixed direct-answer path for unrelated questions.
* Pydantic response validation with `answer`, `sources`, and `confidence`.
* Optional real-LLM generation path with retry logic.
* FastAPI `/ask` endpoint.
* Dockerfile for local container deployment.
* README architecture and example transcripts.
