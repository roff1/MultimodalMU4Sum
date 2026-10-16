"""Teacher inference: images (+ record text) -> validated JSON of aspect summaries.

The backend is a tiny interface so the retry/validation logic is independent of Hugging Face
(and unit-testable without a GPU).
"""
import json
from typing import Dict, List, Protocol, Sequence, Tuple

from omegaconf import DictConfig

from mu4sum.preprocessing.prompts import ASPECT_SUMMARIES_KEY


class TeacherBackend(Protocol):
    """Interface of a text generator that takes images plus a prompt and returns raw text."""

    def generate(self, images: list, prompt: str, *,
                 temperature: float, top_p: float, max_new_tokens: int) -> str: ...


class TeacherFailure(RuntimeError):
    """All attempts for one record produced unparsable / invalid JSON."""

    def __init__(self, errors: List[str]):
        super().__init__(f"{len(errors)} attempts failed: {errors[-1] if errors else ''}")
        self.errors = errors


def _non_empty_text(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} is not a non-empty string")
    return value.strip()


def parse_teacher_json(text: str, keys: Sequence[str]) -> Dict[str, str]:
    """Extract the first JSON object in `text` and return its aspect summaries.

    The object must hold `aspect_summaries` with every aspect as a non-empty string. Extra keys are ignored.
    Raises ValueError describing the first problem found."""
    decoder = json.JSONDecoder()
    obj = None
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            obj = candidate
            break
    if obj is None:
        raise ValueError("no valid JSON object found")

    block = obj.get(ASPECT_SUMMARIES_KEY)
    if not isinstance(block, dict):
        raise ValueError(f"missing '{ASPECT_SUMMARIES_KEY}' object")
    missing = [k for k in keys if k not in block]
    if missing:
        raise ValueError(f"missing aspects: {missing}")
    return {k: _non_empty_text(block[k], f"aspect '{k}'") for k in keys}


def attempt_temperature(gen: DictConfig, attempt: int) -> float:
    """Incremental temperature schedule: base + attempt * step, capped."""
    return min(float(gen.base_temperature) + attempt * float(gen.temperature_step), float(gen.max_temperature))


def generate_aspect_summaries(backend: TeacherBackend, images: list, prompt: str,
                              keys: Sequence[str], gen: DictConfig) -> Tuple[Dict[str, str], dict]:
    """Query the teacher until it returns a structurally valid JSON (1 + max_retries attempts).

    Returns (aspect summaries, {"attempts", "temperature"})."""
    errors: List[str] = []
    for attempt in range(int(gen.max_retries) + 1):
        temperature = attempt_temperature(gen, attempt)
        raw = backend.generate(images, prompt, temperature=temperature,
                               top_p=float(gen.top_p), max_new_tokens=int(gen.max_new_tokens))
        print("[ROFF1] prompt", prompt)
        print("[ROFF1] output", raw)
        try:
            summaries = parse_teacher_json(raw, keys)
        except ValueError as e:
            errors.append(f"attempt {attempt} (T={temperature:.2f}): {e}")
            continue
        return summaries, {"attempts": attempt + 1, "temperature": temperature}
    raise TeacherFailure(errors)


class HFTeacher:
    """Teacher backed by any Hugging Face VLM described by a model preset."""

    def __init__(self, model_cfg: DictConfig):
        from mu4sum.models.loader import load_model, load_processor

        self.model_cfg = model_cfg
        self.processor = load_processor(model_cfg)
        self.model = load_model(model_cfg)
        self.model.eval()
        self.nested = bool(model_cfg.processor.nested_images) if model_cfg.get("processor") else False

    def generate(self, images, prompt, *, temperature, top_p, max_new_tokens) -> str:
        """Return only the newly generated text; temperature 0 means greedy decoding."""
        import torch

        from mu4sum.models.chat import processor_inputs, render_prompt

        text = render_prompt(self.processor, prompt, len(images))
        inputs = processor_inputs(self.processor, [text], [images], self.nested)
        inputs = inputs.to(next(self.model.parameters()).device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 \
            else {"do_sample": False}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
