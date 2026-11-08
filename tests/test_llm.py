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
    bot = object.__new__(LLM)
    bot.client = _Cap()
    bot.model, bot.calls = "m", 0
    return bot


def test_schema_goes_to_format_structured_outputs():
    bot = _llm()
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    out = bot.chat_json("s", "p", schema=schema)
    assert out == {"ok": True}
    assert bot.client.kw["format"] == schema


def test_json_generation_is_bounded_by_num_predict():
    # decompose/judge replies are small; a runaway generation must not hold the
    # runner's only slot for minutes on a CPU-bound box
    bot = _llm()
    bot.chat_json("s", "p")
    assert bot.client.kw["options"]["num_predict"] == JSON_NUM_PREDICT


def test_prose_calls_stay_unbounded_and_unformatted():
    bot = _llm()
    bot.chat("s", "p", json_mode=False, temperature=0.2)
    assert "format" not in bot.client.kw
    assert "num_predict" not in bot.client.kw.get("options", {})
