"""`python -m harness.agent` — start one worker (see compose.yaml `agent` service)."""
from .worker import Agent

if __name__ == "__main__":
    Agent().run()
