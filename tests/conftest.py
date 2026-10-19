import pytest

from falcon.core.config import settings


@pytest.fixture(autouse=True)
def _no_paid_classifier(monkeypatch):
    """Unit tests never spend a token: the guardrail's model classifier is off (its redaction and pattern layers still run).
    The model classifier is measured by `python -m falcon.eval.guardrails` against the labelled message set instead."""
    monkeypatch.setattr(settings, "guardrail_llm", False)
    monkeypatch.setattr(settings, "review_citations", False)  # the citation check is a review-model call; its unit test stubs the model
