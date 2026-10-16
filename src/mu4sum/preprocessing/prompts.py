"""Prompt construction. Teacher and student prompts are built the same way: a text template with placeholders,
filled from one record (`text`, `image_types`, `image_paths`) and the aspects of the dataset preset.

  teacher template  `prompt.teacher_prompt` (dataset preset)   {text} {image_types} {num_images} {json_example}
  student template  `student_prompt` (model preset)            {text} {image_types} {num_images} {json_example}

`{json_example}` is built the same way for both: the JSON structure of the answer, one description per aspect.

`{image_types}` is the comma-separated list of the image types. It is only available when the dataset sets
`prompt.image_types: true`; otherwise every line containing it is dropped, so put it on a line of its own.
"""
import json
import re
from typing import List, Optional, Sequence

from omegaconf import DictConfig

# Key of the JSON object that teacher and student must return; identical for every dataset.
ASPECT_SUMMARIES_KEY = "aspect_summaries"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def aspect_keys(cfg: DictConfig) -> List[str]:
    """Ordered aspect names defined by the dataset preset."""
    return list(cfg.aspects.names)


def image_types_enabled(cfg: DictConfig) -> bool:
    """Whether prompts mention the type of each image (`prompt.image_types`)."""
    return bool(cfg.prompt.get("image_types"))


def _fill(template: str, where: str, **values: Optional[str]) -> str:
    """Replace {name} placeholders line by line; a line using a placeholder whose value is None is dropped.

    Unknown names fail loudly (other braces are left untouched). Inserted values are never re-scanned."""
    lines = []
    for line in template.split("\n"):
        names = _PLACEHOLDER.findall(line)
        for name in names:
            if name not in values:
                raise ValueError(f"Unknown placeholder {{{name}}} in {where}; allowed: {sorted(values)}")
        if any(values[name] is None for name in names):
            continue
        lines.append(_PLACEHOLDER.sub(lambda m: values[m.group(1)], line))
    return "\n".join(lines)


def _require(template: str, placeholder: str, where: str) -> None:
    if "{" + placeholder + "}" not in template:
        raise ValueError(f"{where} must contain the {{{placeholder}}} placeholder")


def _render(cfg: DictConfig, template: str, where: str, text: str, image_types: Sequence[str],
            image_paths: Sequence[str], **extra: str) -> str:
    types = ", ".join(t for t in image_types if t) if image_types_enabled(cfg) else ""
    return _fill(template.strip(), where, text=text or "", image_types=types or None,
                 num_images=str(len(image_paths)), **extra)


def _json_example(cfg: DictConfig, keys: Sequence[str]) -> str:
    """JSON skeleton shown to teacher and student: `aspect_summaries` with one description per aspect."""
    example = {ASPECT_SUMMARIES_KEY: {key: cfg.aspects.descriptions[key] for key in keys}}
    return json.dumps(example, indent=2, ensure_ascii=False)


def build_teacher_prompt(cfg: DictConfig, text: str, image_types: Sequence[str], image_paths: Sequence[str]) -> str:
    """Prompt asking the teacher for the JSON with the aspect summaries."""
    template = cfg.prompt.teacher_prompt
    _require(template, "json_example", "prompt.teacher_prompt")
    return _render(cfg, template, "prompt.teacher_prompt", text, image_types, image_paths,
                   json_example=_json_example(cfg, aspect_keys(cfg)))


def build_student_prompt(cfg: DictConfig, text: str, image_types: Sequence[str], image_paths: Sequence[str],
                         aspects: Optional[Sequence[str]] = None) -> str:
    """Prompt of the model being fine-tuned / unlearned / evaluated, asking for `aspects` (all when None)."""
    template = cfg.get("student_prompt")
    if not template:
        raise ValueError("No `student_prompt` in the model preset (the prompt the model receives as a student)")
    _require(template, "json_example", "student_prompt")
    keys = aspect_keys(cfg) if aspects is None else list(aspects)
    return _render(cfg, template, "student_prompt", text, image_types, image_paths,
                   json_example=_json_example(cfg, keys))
