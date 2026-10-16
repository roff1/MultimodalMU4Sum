import json

from mu4sum.preprocessing.answers import format_answer, parse_answer

ASPECTS = ["heart_and_mediastinum", "lungs_and_pleura", "impression"]
SUMMARIES = {"heart_and_mediastinum": "Heart is normal.", "lungs_and_pleura": "Lungs are clear.",
             "impression": "No acute disease."}


def test_format_answer_is_the_teacher_json_and_spans_cover_each_aspect():
    text, spans = format_answer(SUMMARIES, ASPECTS)
    assert json.loads(text) == {"aspect_summaries": SUMMARIES}
    assert [a for a, _, _ in spans] == ASPECTS
    for aspect, start, end in spans:
        entry = json.dumps(aspect) + ": " + json.dumps(SUMMARIES[aspect])
        assert text[start:end].startswith(entry) and text[start:end][len(entry):] in (",", "}}")
    assert spans[0][1] == text.index('"heart_and_mediastinum"') and spans[-1][2] == len(text)
    only, spans = format_answer(SUMMARIES, ["impression"])
    assert json.loads(only) == {"aspect_summaries": {"impression": "No acute disease."}}
    assert spans == [("impression", only.index('"impression"'), len(only))]


def test_parse_answer_roundtrip_and_robustness():
    text, _ = format_answer(SUMMARIES, ASPECTS)
    assert parse_answer(text, ASPECTS) == SUMMARIES
    assert parse_answer(text, ["impression"]) == {"impression": "No acute disease."}  # extra entries: ignored
    tricky, _ = format_answer({"impression": 'He said "no". Line\nbreak'}, ["impression"])
    assert parse_answer(tricky, ["impression"]) == {"impression": 'He said "no". Line break'}


def test_parse_answer_accepts_text_around_the_json_and_truncated_generations():
    text, _ = format_answer(SUMMARIES, ASPECTS)
    assert parse_answer("Sure!\n```json\n" + text + "\n```", ASPECTS) == SUMMARIES
    cut = text[:text.index('"impression"') + 20]  # generation stopped inside the last summary
    assert parse_answer(cut, ASPECTS) == {k: v for k, v in SUMMARIES.items() if k != "impression"}


def test_parse_answer_matches_by_name_even_out_of_order_or_misspelled():
    text = '{"aspect_summaries": {"Impressions": "Fine.", "lungs & pleura": "Clear.", "heart and mediastinum": "Normal heart."}}'
    assert parse_answer(text, ASPECTS) == {"impression": "Fine.", "lungs_and_pleura": "Clear.",
                                           "heart_and_mediastinum": "Normal heart."}


def test_parse_answer_pairs_leftover_entries_by_similarity_and_leaves_missing_aspects_out():
    text = '{"aspect_summaries": {"chest lung pleura": "Clear.", "impression": "Fine."}}'
    assert parse_answer(text, ASPECTS) == {"impression": "Fine.", "lungs_and_pleura": "Clear."}
    assert parse_answer("", ASPECTS) == {} and parse_answer("just one line", ASPECTS) == {}
    assert parse_answer('{"aspect_summaries": {"impression": ""}}', ASPECTS) == {}
