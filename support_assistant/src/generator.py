import os
import time

from .prompt import STRUCTURED_PROMPT


def generate_answer(question, context):
    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm != "0":
        return f"Based on the retrieved context: {context}"

    prompt = STRUCTURED_PROMPT.format(
        context=context,
        question=question,
    )

    last_error = None

    for attempt in range(2):
        try:
            # Optional real-LLM implementation goes here.
            # The prompt is ready for an LLM provider.
            raise NotImplementedError(
                "Configure a real LLM provider for MOCK_LLM=0."
            )

        except Exception as error:
            last_error = error

            if attempt == 0:
                time.sleep(1)

    raise RuntimeError(
        f"Real LLM generation failed after retry: {last_error}"
    )