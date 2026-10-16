import json

import pytest
from omegaconf import OmegaConf

from mu4sum.preprocessing.prompts import (ASPECT_SUMMARIES_KEY, build_student_prompt,
                                          build_teacher_prompt)
from mu4sum.preprocessing.teacher import (TeacherFailure, attempt_temperature, generate_aspect_summaries,
                                 parse_teacher_json)

KEYS = ["a", "b"]
GEN = OmegaConf.create({"max_new_tokens": 8, "max_retries": 3, "base_temperature": 0.1,
                        "temperature_step": 0.3, "max_temperature": 1.0, "top_p": 0.9})


def wrap(summaries, **extra):
    return json.dumps({ASPECT_SUMMARIES_KEY: summaries, **extra})


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs, self.temperatures = list(outputs), []

    def generate(self, images, prompt, *, temperature, top_p, max_new_tokens):
        self.temperatures.append(temperature)
        return self.outputs.pop(0)


def test_parse_variants():
    expected = {"a": "x", "b": "y"}
    assert parse_teacher_json(f'```json\n{wrap({"a": " x ", "b": "y"})}\n```', KEYS) == expected
    assert parse_teacher_json(f'Sure! {wrap({"a": "x", "b": "y"}, extra=1)} hope it helps', KEYS) == expected
    assert parse_teacher_json('{bad} ' + wrap({"a": "x", "b": "y"}), KEYS) == expected


@pytest.mark.parametrize("text", [
    "no json", '[1]',
    '{"a": "x", "b": "y"}',                                   # flat aspects, no container
    wrap({"a": "x"}), wrap({"a": "x", "b": ""}), wrap({"a": "x", "b": ["y"]}),
    json.dumps({ASPECT_SUMMARIES_KEY: ["x", "y"]}),
])
def test_parse_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_teacher_json(text, KEYS)


def test_temperature_schedule_is_incremental_and_capped():
    assert [round(attempt_temperature(GEN, i), 2) for i in range(5)] == [0.1, 0.4, 0.7, 1.0, 1.0]


def test_retry_until_valid_json():
    backend = ScriptedBackend(["garbage", wrap({"a": "x"}), wrap({"a": "x", "b": "y"})])
    summaries, meta = generate_aspect_summaries(backend, [], "p", KEYS, GEN)
    assert summaries == {"a": "x", "b": "y"}
    assert meta["attempts"] == 3 and meta["temperature"] == pytest.approx(0.7)
    assert backend.temperatures == pytest.approx([0.1, 0.4, 0.7])


def test_failure_after_max_retries():
    backend = ScriptedBackend(["nope"] * 10)
    with pytest.raises(TeacherFailure) as e:
        generate_aspect_summaries(backend, [], "p", KEYS, GEN)
    assert len(backend.temperatures) == 4 and len(e.value.errors) == 4  # 1 + max_retries


def _template_json(prompt):
    return json.loads(prompt[prompt.index("\n{\n") + 1:])


def test_teacher_prompt_structure(make_cfg):
    cfg = make_cfg()
    prompt = build_teacher_prompt(cfg, "FINDINGS: clear", ["Frontal", "Lateral"], ["a.png", "b.png"])
    assert "Frontal, Lateral" in prompt and "'FINDINGS: clear'" in prompt and "2 chest X-ray" in prompt
    assert not any(p in prompt for p in ("{text}", "{image_types}", "{num_images}", "{json_example}"))
    assert prompt.index("FINDINGS: clear") < prompt.index("GUIDELINES") < prompt.rindex("\n{\n")

    template = _template_json(prompt)
    assert list(template) == [ASPECT_SUMMARIES_KEY]
    assert list(template[ASPECT_SUMMARIES_KEY]) == list(cfg.aspects.names)
    for key in cfg.aspects.names:
        assert template[ASPECT_SUMMARIES_KEY][key] == cfg.aspects.descriptions[key]


def test_student_prompt_builds_the_same_json_example_for_the_asked_aspects(make_cfg):
    cfg = make_cfg("train", training="ga_gd")
    teacher = _template_json(build_teacher_prompt(cfg, "t", ["Frontal"], ["a.png"]))
    full = build_student_prompt(cfg, "FINDINGS: clear", ["Frontal"], ["a.png"])
    assert "'FINDINGS: clear'" in full and "Frontal" in full and "1 chest X-ray" in full
    assert _template_json(full) == teacher  # all aspects: identical to the teacher's example

    only = build_student_prompt(cfg, "t", ["Frontal"], ["a.png"], ["impression"])
    assert _template_json(only) == {ASPECT_SUMMARIES_KEY: {"impression": cfg.aspects.descriptions["impression"]}}
    assert "lungs_and_pleura" not in only and "{json_example}" not in only


def test_image_types_line_is_dropped_when_disabled_or_unknown(make_cfg):
    off = build_teacher_prompt(make_cfg(overrides=["prompt.image_types=false"]), "r", ["Frontal"], ["a.png"])
    assert "Image types" not in off and "Frontal" not in off
    unknown = build_teacher_prompt(make_cfg(), "r", ["", ""], ["a.png", "b.png"])
    assert "Image types" not in unknown


def test_text_is_shown_only_if_the_prompt_uses_it(make_cfg):
    cfg = make_cfg(overrides=['prompt.teacher_prompt="Describe {num_images} image(s).\n{json_example}"'])
    prompt = build_teacher_prompt(cfg, "SECRET REPORT", ["Frontal"], ["a.png"])
    assert "SECRET REPORT" not in prompt and prompt.startswith("Describe 1 image(s).")


def test_unknown_placeholder_fails_loudly(make_cfg):
    cfg = make_cfg(overrides=['prompt.teacher_prompt="Look at {images}\n{json_example}"'])
    with pytest.raises(ValueError, match=r"\{images\}"):
        build_teacher_prompt(cfg, "r", [], [])


def test_required_placeholders(make_cfg):
    with pytest.raises(ValueError, match="json_example"):
        build_teacher_prompt(make_cfg(overrides=['prompt.teacher_prompt="no example"']), "r", [], [])
    cfg = make_cfg("train", training="ga_gd", overrides=['student_prompt="no example"'])
    with pytest.raises(ValueError, match="json_example"):
        build_student_prompt(cfg, "r", [], [])
