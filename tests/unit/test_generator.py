
from types import SimpleNamespace

import pytest

from src import generator


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError, match="question ne peut pas être vide"):
        generator.generate_answer("", "Contexte documentaire")


def test_generate_answer_rejects_whitespace_question():
    with pytest.raises(ValueError, match="question ne peut pas être vide"):
        generator.generate_answer("   ", "Contexte documentaire")


def test_generate_answer_returns_message_without_context():
    result = generator.generate_answer(
        "Comment configurer DNS ?",
        ""
    )

    assert result == generator.INSUFFICIENT_CONTEXT_MESSAGE


def test_generate_answer_does_not_call_groq_without_context(monkeypatch):
    def fake_client():
        raise AssertionError("Groq ne doit pas être appelé sans contexte")

    monkeypatch.setattr(generator, "get_groq_client", fake_client)

    result = generator.generate_answer(
        "Comment configurer DNS ?",
        "   "
    )

    assert result == generator.INSUFFICIENT_CONTEXT_MESSAGE


def test_generate_answer_with_valid_response(monkeypatch):
    expected_answer = "Pour vider le cache DNS, utilisez ipconfig /flushdns."

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=expected_answer)
                        )
                    ]
                )
            )
        )
    )

    monkeypatch.setattr(
        generator,
        "get_groq_client",
        lambda: fake_client
    )

    result = generator.generate_answer(
        "Comment vider le cache DNS ?",
        "DOC002 : commande ipconfig /flushdns"
    )

    assert result == expected_answer


def test_generate_answer_returns_insufficient_message_for_empty_response(
    monkeypatch
):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=None)
                        )
                    ]
                )
            )
        )
    )

    monkeypatch.setattr(
        generator,
        "get_groq_client",
        lambda: fake_client
    )

    result = generator.generate_answer(
        "Comment vider le cache DNS ?",
        "Contexte documentaire"
    )

    assert result == generator.INSUFFICIENT_CONTEXT_MESSAGE


def test_get_groq_client_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        generator.get_groq_client()


def test_get_groq_client_initializes_groq(monkeypatch):
    captured = {}

    def fake_groq(**kwargs):
        captured.update(kwargs)
        return "fake_groq_client"

    monkeypatch.setenv("GROQ_API_KEY", "test_api_key")
    monkeypatch.setattr(generator, "Groq", fake_groq)

    result = generator.get_groq_client()

    assert result == "fake_groq_client"
    assert captured["api_key"] == "test_api_key"
