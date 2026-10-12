"""Multi-aspect answers, the output format of MU4Sum: one block per aspect, `<aspect name>\\n<summary>`,
blocks separated by a blank line. Used to build training targets and to read generated text back."""
import re
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

# Edit distance below which two aspect names are considered the same (as in the MU4Sum evaluation code).
MAX_NAME_EDIT_DISTANCE = 3

Span = Tuple[str, int, int]  # (aspect, start, end) character offsets of an aspect block inside the answer


def display_name(aspect: str) -> str:
    """Aspect name as written in the answer: 'lungs_and_pleura' -> 'lungs and pleura'."""
    return aspect.strip().replace("_", " ")


def format_answer(summaries: Dict[str, str], aspects: Sequence[str]) -> Tuple[str, List[Span]]:
    """The answer text for `aspects` (in this order) and the character span of each aspect block."""
    parts, spans, pos = [], [], 0
    for aspect in aspects:
        block = f"{display_name(aspect)}\n{summaries[aspect].strip()}"
        spans.append((aspect, pos, pos + len(block)))
        parts.append(block)
        pos += len(block) + 2  # the "\n\n" separator
    return "\n\n".join(parts), spans


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


def _blocks(text: str) -> List[Tuple[str, str]]:
    """(header, summary) pairs of a generated answer; the header is the first line of each blank-line block."""
    pairs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if len(lines) >= 2:
            pairs.append((lines[0], " ".join(lines[1:])))
    return pairs


def parse_answer(text: str, aspects: Sequence[str]) -> Dict[str, str]:
    """Map each requested aspect to the summary the model wrote for it (aspects it did not write are absent).

    A block is assigned to the aspect whose name it repeats (substring or small edit distance); blocks left over
    are paired with the remaining aspects by name similarity, most similar first."""
    blocks, free = _blocks(text), list(aspects)
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
