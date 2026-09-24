from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_valid_question():
    response = client.post(
        "/ask",
        json={
            "question": "Can I cancel my Zepto order?"
        },
    )

    assert response.status_code == 200
    assert "answer" in response.json()
    assert "source" in response.json()


def test_unknown_question():
    response = client.post(
        "/ask",
        json={
            "question": "What is the weather today?"
        },
    )

    assert response.status_code == 200
    assert response.json()["source"] == ""


def test_missing_question():
    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 400


def test_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 400


def test_extra_field():
    response = client.post(
        "/ask",
        json={
            "question": "Can I cancel my Zepto order?",
            "name": "test",
        },
    )

    assert response.status_code == 400