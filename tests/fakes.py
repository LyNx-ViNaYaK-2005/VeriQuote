"""Deterministic provider fakes for tests only; never imported by the application."""
from types import SimpleNamespace
import json

import numpy as np


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        # Deliberately return unnormalized float64 data to exercise the
        # embedding adapter's normalization and FAISS dtype guarantee.
        return np.array([[1.0, 0.2, 0.1] for _ in texts], dtype=np.float64)


class FakeGroq:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        request = json.loads(kwargs["messages"][1]["content"])
        passage = request["evidence"][0]
        payload = {"claims": [{"text": passage["text"], "evidence": [{"id": passage["id"], "quote": passage["text"]}]}]}
        if "weather" in request["question"].lower():
            payload = {"claims": []}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))])
