"""Tests for the Ollama client, with urllib replaced so no real Ollama is needed."""

import io
import json
import urllib.request

from app.extractor import OllamaLLM


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_ollama_sends_json_request_and_returns_content(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({"message": {"content": '{"ok": true}'}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    llm = OllamaLLM(model="qwen2.5:7b", base_url="http://localhost:11434/")

    assert llm.complete("system text", "user text") == '{"ok": true}'
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "qwen2.5:7b"
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0] == {"role": "system", "content": "system text"}
