"""On-disk cache: cache/<sha256>/ holds one file per pipeline layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

DEFAULT_ROOT = Path("cache")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Cache:
    def __init__(self, key: str, root: Path = DEFAULT_ROOT):
        self.key = key
        self.dir = Path(root) / key
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def load_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text())

    def save_json(self, name: str, data: Any) -> None:
        self.path(name).write_text(json.dumps(data, indent=1))

    def get_or_compute(self, name: str, fn: Callable[[], Any], refresh: bool = False) -> Any:
        """Return cached JSON for `name`, computing and storing it on a miss."""
        if self.has(name) and not refresh:
            return self.load_json(name)
        data = fn()
        self.save_json(name, data)
        return data
