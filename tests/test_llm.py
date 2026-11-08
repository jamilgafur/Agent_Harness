"""LLM client: structured-output format, bounded JSON generations."""
from harness.llm import JSON_NUM_PREDICT, LLM


class _Cap:
    """Fake ollama.Client: captures kwargs, replies valid JSON."""

    def __init__(self):
        self.kw = None

    def chat(self, model, messages, **kw):
        self.kw = kw
        return {"message": {"content": '{"ok": true}'}}


def _llm():
    l = object.__new__(LLM)
    l.client = _Cap()
    l.model, l.calls = "m", 0
    return l


def test_schema_goes_to_format_structured_outputs():
    l = _llm()
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    out = l.chat_json("s", "p", schema=schema)
    assert out == {"ok": True}
    assert l.client.kw["format"] == schema


def test_json_generation_is_bounded_by_num_predict():
    # decompose/judge replies are small; a runaway generation must not hold the
    # runner's only slot for minutes on a CPU-bound box
    l = _llm()
    l.chat_json("s", "p")
    assert l.client.kw["options"]["num_predict"] == JSON_NUM_PREDICT


def test_prose_calls_stay_unbounded_and_unformatted():
    l = _llm()
    l.chat("s", "p", json_mode=False, temperature=0.2)
    assert "format" not in l.client.kw
    assert "num_predict" not in l.client.kw.get("options", {})
