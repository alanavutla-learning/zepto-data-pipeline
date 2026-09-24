from transformers import pipeline


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def create_generator():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
    )


def generate_answer(question, retrieved_documents):
    generator = create_generator()

    context = "\n\n".join(
        f"Source: {document['source']}\n"
        f"{document['content']}"
        for document in retrieved_documents
    )

    prompt = f"""You are a strict customer support assistant.

Use ONLY the policy information provided below.

Rules:
- Do not add information that is not in the policy.
- Do not invent contact details, email addresses, procedures, fees, or conditions.
- Give a short direct answer.
- If the policy does not answer the question, say:
"I don't have enough information in the available policy documents."

Policy information:
{context}

Customer question:
{question}

Answer:"""

    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        return_full_text=False,
    )

    return result[0]["generated_text"].strip()


if __name__ == "__main__":
    question = "Can I cancel my Zepto order?"

    documents = [
        {
            "source": "doc_05.txt",
            "content": (
                "Orders can be cancelled free of cost any time before "
                "the order status changes to 'Packed', typically within "
                "the first 2 minutes of placing the order. Once an order "
                "has been packed, it can no longer be cancelled through "
                "the app."
            ),
        }
    ]

    answer = generate_answer(question, documents)

    print("Question:")
    print(question)

    print("\nGenerated answer:")
    print(answer)