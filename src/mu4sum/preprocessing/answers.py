"""Multi-aspect answers: the JSON object the student is asked for (the same structure as the teacher's),
`{"aspect_summaries": {"<aspect name>": "<summary>", ...}}`. Used to build training targets and to read generated
text back (also when the generation was cut off or is not valid JSON)."""
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

from mu4sum.preprocessing.prompts import ASPECT_SUMMARIES_KEY

# Edit distance below which two aspect names are considered the same (as in the MU4Sum evaluation code).
MAX_NAME_EDIT_DISTANCE = 3

Span = Tuple[str, int, int]  # (aspect, start, end) character offsets of an aspect inside the answer

_PREFIX = '{' + json.dumps(ASPECT_SUMMARIES_KEY) + ': {'
_SUFFIX = '}}'
_STRING_ENTRY = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def format_answer(summaries: Dict[str, str], aspects: Sequence[str]) -> Tuple[str, List[Span]]:
    """The answer text for `aspects` (in this order) and the character span of each aspect.

    A span is the `"aspect": "summary"` entry plus the delimiter that closes it (`,` or the final `}}`), so a token
    fusing the closing quote with that delimiter belongs to the aspect it ends."""
    entries = [f"{json.dumps(a, ensure_ascii=False)}: {json.dumps(summaries[a].strip(), ensure_ascii=False)}"
               for a in aspects]
    spans, pos = [], len(_PREFIX)
    for i, (aspect, entry) in enumerate(zip(aspects, entries)):
        last = i == len(entries) - 1
        spans.append((aspect, pos, pos + len(entry) + (len(_SUFFIX) if last else 1)))
        pos += len(entry) + 2  # the ", " separator
    return _PREFIX + ", ".join(entries) + _SUFFIX, spans


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _same_aspect(header: str, aspect: str) -> bool:
    h, a = _norm(header), _norm(aspect)
    return bool(h) and (h in a or a in h or _edit_distance(h, a) <= MAX_NAME_EDIT_DISTANCE)


def _pairs(text: str) -> List[Tuple[str, str]]:
    """(name, summary) pairs of every complete `"name": "text"` entry of a generated answer, in order."""
    pairs = []
    for name, value in _STRING_ENTRY.findall(text):
        try:
            name, value = json.loads(f'"{name}"'), json.loads(f'"{value}"')
        except json.JSONDecodeError:
            continue
        if value.strip():
            pairs.append((name, " ".join(value.split())))
    return pairs


def parse_answer(text: str, aspects: Sequence[str]) -> Dict[str, str]:
    """Map each requested aspect to the summary the model wrote for it (aspects it did not write are absent).

    An entry is assigned to the aspect whose name it repeats (substring or small edit distance); entries left over
    are paired with the remaining aspects by name similarity, most similar first."""
    blocks, free = _pairs(text), list(aspects)
    found: Dict[str, str] = {}
    unmatched = []
    for header, summary in blocks:
        match = next((a for a in free if _same_aspect(header, a)), None)
        if match is None:
            unmatched.append((header, summary))
        else:
            found[match] = summary
            free.remove(match)
    while unmatched and free:
        score, i, aspect = max(((SequenceMatcher(None, _norm(h), _norm(a)).ratio(), i, a)
                                for i, (h, _) in enumerate(unmatched) for a in free), key=lambda t: t[0])
        found[aspect] = unmatched.pop(i)[1]
        free.remove(aspect)
    return found
