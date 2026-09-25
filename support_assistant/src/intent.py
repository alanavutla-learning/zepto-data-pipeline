import os


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(question):
    mock_llm = os.getenv("MOCK_LLM", "1")

    question_lower = question.lower()

    if any(keyword in question_lower for keyword in POLICY_KEYWORDS):
        return "policy_question"

    return "general_question"
