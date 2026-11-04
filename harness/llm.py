"""Ollama client with JSON-mode and a repair round-trip.

Small local models mangle JSON; this is the single failure isolation point.
"""
import json
from typing import Any, Dict, Optional

import ollama


class LLM:
    def __init__(self, url: str, model: str, timeout: float = 300.0):
        self.client = ollama.Client(host=url, timeout=timeout)
        self.model = model
        self.calls = 0

    def chat(self, system: str, prompt: str, json_mode: bool = True) -> str:
        self.calls += 1
        kw = {"format": "json"} if json_mode else {}
        r = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            **kw,
        )
        return r["message"]["content"]

    def chat_json(self, system: str, prompt: str, repairs: int = 2) -> Optional[Dict[str, Any]]:
        """Ask for JSON; on parse failure, feed the error back and let the model fix it."""
        err = ""
        for _ in range(1 + repairs):
            out = self.chat(system, prompt + err)
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
