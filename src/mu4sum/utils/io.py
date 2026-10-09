"""JSON / JSONL read-write helpers (UTF-8, parent directories created on write)."""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_json(path) -> Any:
    """Load and return the JSON document at path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, obj: Any) -> None:
    """Write via temp file + rename so a Colab disconnect never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_jsonl(path) -> Iterator[dict]:
    """Lazily yield one record per non-blank line; yields nothing if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path, record: dict) -> None:
    """Append one record as a line and flush, creating the file and parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path, records: Iterable[dict]) -> None:
    """Overwrite path with one JSON record per line (not atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
