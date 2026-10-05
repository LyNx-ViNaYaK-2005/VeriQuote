"""Deterministic provider fakes for tests only; never imported by the application."""
from types import SimpleNamespace
import json



class FakeJinaClient:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json):
        self.calls.append(json)
        vectors = [[1.0, 0.2, 0.1] for _ in json["input"]]
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"data": [{"index": i, "embedding": value}
                                                       for i, value in enumerate(vectors)]})


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
