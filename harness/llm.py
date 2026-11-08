"""Ollama client with JSON-mode and a repair round-trip.

Small local models mangle JSON; this is the single failure isolation point.
"""
import json
import os
from typing import Any

import ollama

# Bound structured replies (decompose/judge): a runaway JSON generation on a
# CPU-bound box holds the runner's only slot for minutes — cap it instead.
# Prose (writers) stays unbounded. 0 = unbounded.
JSON_NUM_PREDICT = int(os.environ.get("LLM_JSON_NUM_PREDICT", "768"))


class LLM:
    def __init__(self, url: str, model: str, timeout: float = 300.0):
        self.client = ollama.Client(host=url, timeout=timeout)
        self.model = model
        self.calls = 0

    def chat(self, system: str, prompt: str, json_mode: bool = True,
             model: str = "", temperature: float | None = None,
             schema: dict[str, Any] | None = None,
             num_predict: int = 0) -> str:
        """One chat call. model/temperature override per run (run params); the
        default model comes from the constructor (env MODEL). A JSON schema goes
        to Ollama's `format` (structured outputs) instead of prose-described JSON.
        num_predict > 0 caps the reply length."""
        self.calls += 1
        kw: dict[str, Any] = {}
        if json_mode:
            kw["format"] = schema if schema else "json"
        opts: dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = temperature
        if num_predict > 0:
            opts["num_predict"] = num_predict
        if opts:
            kw["options"] = opts
        r = self.client.chat(
            model=model or self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            **kw,
        )
        return r["message"]["content"]

    def chat_json(self, system: str, prompt: str, repairs: int = 2,
                  model: str = "", temperature: float | None = None,
                  schema: dict[str, Any] | None = None,
                  num_predict: int = JSON_NUM_PREDICT) -> dict[str, Any] | None:
        """Ask for JSON; on parse failure, feed the error back and let the model fix it."""
        err = ""
        for _ in range(1 + repairs):
            out = self.chat(system, prompt + err, model=model, temperature=temperature,
                            schema=schema, num_predict=num_predict)
            try:
                return json.loads(self._extract(out))
            except (json.JSONDecodeError, ValueError) as e:
                err = f"\n\nYour previous reply was not valid JSON ({e}). Reply with corrected JSON only."
        return None

    @staticmethod
    def _extract(text: str) -> str:
        """Tolerate markdown fences / prose around the JSON object."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1].lstrip("json\n ")
            text = text.rsplit("```", 1)[0]
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
        return text[start:]

    def ping(self) -> float:
        """Round-trip seconds for a trivial call — used by the agent autosizer."""
        import time
        t = time.time()
        self.chat("Reply with the single word: ok", "ping", json_mode=False)
        return time.time() - t
