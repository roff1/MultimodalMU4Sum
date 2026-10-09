import json

import pytest
from omegaconf import OmegaConf

from mu4sum.data.prompts import (ASPECT_SUMMARIES_KEY, EXPANDED_TEXT_KEY, build_student_prompt,
                                 build_teacher_prompt, describe_images)
from mu4sum.data.teacher import (TeacherFailure, attempt_temperature, generate_aspect_summaries,
                                 parse_teacher_json)

KEYS = ["a", "b"]
GEN = OmegaConf.create({"max_new_tokens": 8, "max_retries": 3, "base_temperature": 0.1,
                        "temperature_step": 0.3, "max_temperature": 1.0, "top_p": 0.9})


def wrap(summaries, expanded=None, **extra):
    obj = {ASPECT_SUMMARIES_KEY: summaries, **extra}
    if expanded is not None:
        obj[EXPANDED_TEXT_KEY] = expanded
    return json.dumps(obj)


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs, self.temperatures = list(outputs), []

    def generate(self, images, system, prompt, *, temperature, top_p, max_new_tokens):
        self.temperatures.append(temperature)
        return self.outputs.pop(0)


def test_parse_variants():
    expected = ({"a": "x", "b": "y"}, None)
    assert parse_teacher_json(f'```json\n{wrap({"a": " x ", "b": "y"})}\n```', KEYS) == expected
    assert parse_teacher_json(f'Sure! {wrap({"a": "x", "b": "y"}, extra=1)} hope it helps', KEYS) == expected
    assert parse_teacher_json('{bad} ' + wrap({"a": "x", "b": "y"}), KEYS) == expected


def test_parse_expanded_text_only_when_requested():
    text = wrap({"a": "x", "b": "y"}, " long text ")
    assert parse_teacher_json(text, KEYS, expand=True) == ({"a": "x", "b": "y"}, "long text")
    assert parse_teacher_json(text, KEYS, expand=False)[1] is None
    with pytest.raises(ValueError, match=EXPANDED_TEXT_KEY):
        parse_teacher_json(wrap({"a": "x", "b": "y"}), KEYS, expand=True)


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
    backend = ScriptedBackend(["garbage", wrap({"a": "x"}), wrap({"a": "x", "b": "y"}, "long")])
    summaries, expanded, meta = generate_aspect_summaries(backend, [], None, "p", KEYS, GEN, expand=True)
    assert summaries == {"a": "x", "b": "y"} and expanded == "long"
    assert meta["attempts"] == 3 and meta["temperature"] == pytest.approx(0.7)
    assert backend.temperatures == pytest.approx([0.1, 0.4, 0.7])


def test_failure_after_max_retries():
    backend = ScriptedBackend(["nope"] * 10)
    with pytest.raises(TeacherFailure) as e:
        generate_aspect_summaries(backend, [], None, "p", KEYS, GEN)
    assert len(backend.temperatures) == 4 and len(e.value.errors) == 4  # 1 + max_retries


def test_describe_images():
    assert describe_images(["Frontal", "Lateral"]) == "Image 1 (Frontal) and Image 2 (Lateral)"
    assert describe_images(["Frontal"]) == "Image 1 (Frontal)"
    assert describe_images(["", ""]) == "Image 1 and Image 2"
    assert describe_images(["A", "B", "C"]) == "Image 1 (A), Image 2 (B) and Image 3 (C)"
    assert "image" in describe_images([])


def _template_json(prompt):
    return json.loads(prompt[prompt.index("\n{\n") + 1:])


def test_teacher_prompt_structure(make_cfg):
    cfg = make_cfg()
    prompt = build_teacher_prompt(cfg, "FINDINGS: clear", ["Frontal", "Lateral"])
    assert "Image 1 (Frontal) and Image 2 (Lateral)" in prompt and "'FINDINGS: clear'" in prompt
    assert "{image_info}" not in prompt and "{source_text}" not in prompt
    assert prompt.index("FINDINGS: clear") < prompt.index("OUTPUT RULES") < prompt.rindex("\n{\n")

    template = _template_json(prompt)
    assert list(template) == [EXPANDED_TEXT_KEY, ASPECT_SUMMARIES_KEY]
    assert list(template[ASPECT_SUMMARIES_KEY]) == list(cfg.aspects.names)
    for key in cfg.aspects.names:
        assert template[ASPECT_SUMMARIES_KEY][key] == f"[{cfg.aspects.descriptions[key]}]"
    assert template[EXPANDED_TEXT_KEY] == f"[{cfg.prompt.expansion.description}]"


def test_student_prompt_is_unchanged_by_teacher_settings(make_cfg):
    cfg = make_cfg()
    student = build_student_prompt(cfg, ["impression"])
    assert "Aspect 1/1: impression" in student and "lungs_and_pleura" not in student


def test_expansion_can_be_disabled(make_cfg):
    cfg = make_cfg(overrides=["prompt.expansion.enabled=false"])
    template = _template_json(build_teacher_prompt(cfg, "r", ["Frontal"]))
    assert list(template) == [ASPECT_SUMMARIES_KEY]


def test_names_only_mode_hides_aspect_descriptions(make_cfg):
    cfg = make_cfg(overrides=["prompt.mode=names_only"])
    template = _template_json(build_teacher_prompt(cfg, "r", []))
    assert set(template[ASPECT_SUMMARIES_KEY].values()) == {"<summary>"}
    assert "Description" not in build_student_prompt(cfg, ["impression"])


def test_source_text_is_shown_only_if_the_intro_uses_it(make_cfg):
    cfg = make_cfg(overrides=['prompt.teacher_intro="Describe the image(s): {image_info}."'])
    prompt = build_teacher_prompt(cfg, "SECRET REPORT", ["Frontal"])
    assert "SECRET REPORT" not in prompt and "Describe the image(s): Image 1 (Frontal)." in prompt


def test_unknown_intro_placeholder_fails_loudly(make_cfg):
    cfg = make_cfg(overrides=['prompt.teacher_intro="Look at {images}"'])
    with pytest.raises(ValueError, match=r"\{images\}"):
        build_teacher_prompt(cfg, "r", [])


def test_enabled_expansion_requires_a_description(make_cfg):
    cfg = make_cfg(overrides=["prompt.expansion.description="])
    with pytest.raises(ValueError, match="description"):
        build_teacher_prompt(cfg, "r", [])
