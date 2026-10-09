import pytest
from omegaconf import OmegaConf

from mu4sum.data.prompts import build_student_prompt, build_teacher_prompt
from mu4sum.data.teacher import (TeacherFailure, attempt_temperature, generate_aspect_summaries,
                                 parse_aspect_json)

KEYS = ["a", "b"]
GEN = OmegaConf.create({"max_new_tokens": 8, "max_retries": 3, "base_temperature": 0.1,
                        "temperature_step": 0.3, "max_temperature": 1.0, "top_p": 0.9})


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs, self.temperatures = list(outputs), []

    def generate(self, images, system, prompt, *, temperature, top_p, max_new_tokens):
        self.temperatures.append(temperature)
        return self.outputs.pop(0)


def test_parse_variants():
    assert parse_aspect_json('```json\n{"a": " x ", "b": "y"}\n```', KEYS) == {"a": "x", "b": "y"}
    assert parse_aspect_json('Sure! {"a": "x", "b": "y", "extra": 1} hope it helps', KEYS) == {"a": "x", "b": "y"}
    assert parse_aspect_json('{bad} {"a": "x", "b": "y"}', KEYS) == {"a": "x", "b": "y"}


@pytest.mark.parametrize("text", ["no json", '{"a": "x"}', '{"a": "x", "b": ""}', '{"a": "x", "b": ["y"]}', '[1]'])
def test_parse_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_aspect_json(text, KEYS)


def test_temperature_schedule_is_incremental_and_capped():
    assert [round(attempt_temperature(GEN, i), 2) for i in range(5)] == [0.1, 0.4, 0.7, 1.0, 1.0]


def test_retry_until_valid_json():
    backend = ScriptedBackend(["garbage", '{"a": "x"}', '{"a": "x", "b": "y"}'])
    summaries, meta = generate_aspect_summaries(backend, [], None, "p", KEYS, GEN)
    assert summaries == {"a": "x", "b": "y"}
    assert meta["attempts"] == 3 and meta["temperature"] == pytest.approx(0.7)
    assert backend.temperatures == pytest.approx([0.1, 0.4, 0.7])


def test_failure_after_max_retries():
    backend = ScriptedBackend(["nope"] * 10)
    with pytest.raises(TeacherFailure) as e:
        generate_aspect_summaries(backend, [], None, "p", KEYS, GEN)
    assert len(backend.temperatures) == 4 and len(e.value.errors) == 4  # 1 + max_retries


def test_incremental_prompt_lists_every_aspect_with_description(make_cfg):
    cfg = make_cfg()
    prompt = build_teacher_prompt(cfg, "FINDINGS: clear", ["Frontal", "Lateral"])
    for i, key in enumerate(cfg.aspects.names, start=1):
        assert f"Aspect {i}/4: {key}" in prompt
        assert cfg.aspects.descriptions[key] in prompt
    assert "Frontal, Lateral views" in prompt and "FINDINGS: clear" in prompt and '"impression"' in prompt

    student = build_student_prompt(cfg, ["impression"])
    assert "Aspect 1/1: impression" in student and "lungs_and_pleura" not in student


def test_names_only_mode_and_report_toggle(make_cfg):
    cfg = make_cfg(overrides=["prompt.mode=names_only", "prompt.teacher_use_report=false"])
    prompt = build_teacher_prompt(cfg, "SECRET REPORT", [])
    assert "- impression" in prompt and "Description" not in prompt and "SECRET REPORT" not in prompt
