"""Unit tests: DAG logic (pure) + core API (in-memory, sandbox mocked)."""
import os
import sys

os.environ["HARNESS_CONFIG"] = '{"data_dir": "/tmp/harness-test", "sandbox_url": "http://unused"}'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
