"""Prompt construction driven entirely by the dataset preset (`aspects` + `prompt` sections)."""
import json
import re
from typing import List, Sequence

from omegaconf import DictConfig

# Keys of the JSON object the teacher must return; identical for every dataset.
EXPANDED_TEXT_KEY = "expanded_text"
ASPECT_SUMMARIES_KEY = "aspect_summaries"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")

_OUTPUT_RULES = f"""OUTPUT RULES:
1. FORMAT: Output ONLY a valid JSON object. Do not include any conversational text, markdown formatting (like ```json), or explanations outside the JSON block.
2. FLAT TEXT VALUES ONLY: Every value must be a single text string; the only nested object allowed is '{ASPECT_SUMMARIES_KEY}', whose values are text strings too.
3. Use exactly the following JSON keys and structure:"""


def aspect_keys(cfg: DictConfig) -> List[str]:
    """Ordered aspect names defined by the dataset preset."""
    return list(cfg.aspects.names)


def expansion_enabled(cfg: DictConfig) -> bool:
    """Whether the teacher must also expand the dataset text (`prompt.expansion.enabled`)."""
    expansion = cfg.prompt.get("expansion")
    return bool(expansion and expansion.get("enabled"))


def _show_descriptions(cfg: DictConfig) -> bool:
    mode = cfg.prompt.mode
    if mode not in ("incremental", "names_only"):
        raise ValueError(f"Unknown prompt.mode '{mode}' (expected 'incremental' or 'names_only')")
    return mode == "incremental"


def _aspect_block(cfg: DictConfig, keys: Sequence[str]) -> str:
    if _show_descriptions(cfg):
        # Built aspect by aspect: "Aspect i/N: <key>" followed by its description.
        lines = []
        for i, key in enumerate(keys, start=1):
            lines.append(f"Aspect {i}/{len(keys)}: {key}")
            lines.append(f"  Description: {cfg.aspects.descriptions[key]}")
        return "\n".join(lines)
    return "\n".join(f"- {key}" for key in keys)


def _fill(template: str, where: str, **values: str) -> str:
    """Replace {name} placeholders; unknown names fail loudly (other braces are left untouched)."""
    def replace(match):
        name = match.group(1)
        if name not in values:
            raise ValueError(f"Unknown placeholder {{{name}}} in {where}; allowed: {sorted(values)}")
        return values[name]

    return _PLACEHOLDER.sub(replace, template)


def describe_images(views: Sequence[str]) -> str:
    """'Image 1 (Frontal) and Image 2 (Lateral)'; one entry per image, view omitted when unknown."""
    if not views:
        return "the provided image(s)"
    items = [f"Image {i} ({v})" if v else f"Image {i}" for i, v in enumerate(views, start=1)]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _output_template(cfg: DictConfig, keys: Sequence[str]) -> str:
    """JSON skeleton shown to the teacher: optional expanded text + one slot per aspect."""
    show = _show_descriptions(cfg)

    def slot(key: str) -> str:
        return f"[{cfg.aspects.descriptions[key]}]" if show else "<summary>"

    template = {}
    if expansion_enabled(cfg):
        description = cfg.prompt.expansion.get("description")
        if not description:
            raise ValueError("prompt.expansion.enabled is true but prompt.expansion.description is empty")
        template[EXPANDED_TEXT_KEY] = f"[{description}]"
    template[ASPECT_SUMMARIES_KEY] = {key: slot(key) for key in keys}
    return json.dumps(template, indent=2, ensure_ascii=False)


def build_student_prompt(cfg: DictConfig, keys: Sequence[str]) -> str:
    """Prompt for the model being fine-tuned / unlearned / evaluated (images are the input)."""
    return f"{cfg.prompt.student_task.strip()}\n\n{_aspect_block(cfg, keys)}"


def build_teacher_prompt(cfg: DictConfig, source_text: str, views: Sequence[str]) -> str:
    """Teacher prompt: dataset intro ({image_info}, {source_text}) + output rules + JSON template.

    `views` has one entry per image of the study (empty string when the view is unknown)."""
    intro = _fill(cfg.prompt.teacher_intro.strip(), "prompt.teacher_intro",
                  image_info=describe_images(views), source_text=source_text or "")
    return "\n\n".join([intro, _OUTPUT_RULES, _output_template(cfg, aspect_keys(cfg))])
