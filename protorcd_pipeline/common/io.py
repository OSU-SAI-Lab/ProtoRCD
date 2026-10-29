"""I/O helpers for JSON, JSONL, CSV, and optional YAML files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=indent)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_yaml(path: str | Path) -> Any:
    import yaml

    with Path(path).open("r") as f:
        return yaml.safe_load(f)

