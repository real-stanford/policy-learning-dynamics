from __future__ import annotations

import json


class JsonLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._handle = None

    def __enter__(self):
        self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        assert self._handle is not None
        self._handle.close()
        self._handle = None

    def log(self, payload: dict) -> None:
        assert self._handle is not None
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()
