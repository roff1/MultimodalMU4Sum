from mu4sum.preprocessing.answers import display_name, format_answer, parse_answer

ASPECTS = ["heart_and_mediastinum", "lungs_and_pleura", "impression"]
SUMMARIES = {"heart_and_mediastinum": "Heart is normal.", "lungs_and_pleura": "Lungs are clear.",
             "impression": "No acute disease."}


def test_format_answer_blocks_and_spans():
    text, spans = format_answer(SUMMARIES, ASPECTS)
    assert text == ("heart and mediastinum\nHeart is normal.\n\nlungs and pleura\nLungs are clear.\n\n"
                    "impression\nNo acute disease.")
    assert [a for a, _, _ in spans] == ASPECTS
    for aspect, start, end in spans:
        assert text[start:end] == f"{display_name(aspect)}\n{SUMMARIES[aspect]}"
    only, spans = format_answer(SUMMARIES, ["impression"])
    assert only == "impression\nNo acute disease." and spans == [("impression", 0, len(only))]


def test_parse_answer_roundtrip_and_robustness():
    text, _ = format_answer(SUMMARIES, ASPECTS)
    assert parse_answer(text, ASPECTS) == SUMMARIES
    assert parse_answer(text, ["impression"]) == {"impression": "No acute disease."}  # extra block: unmatched, ignored


def test_parse_answer_matches_by_name_even_out_of_order_or_misspelled():
    text = "Impressions\nFine.\n\nlungs & pleura\nClear.\n\nheart and mediastinum\nNormal heart.\nSecond line."
    assert parse_answer(text, ASPECTS) == {"impression": "Fine.", "lungs_and_pleura": "Clear.",
                                           "heart_and_mediastinum": "Normal heart. Second line."}


def test_parse_answer_pairs_leftover_blocks_by_similarity_and_leaves_missing_aspects_out():
    text = "chest lung pleura\nClear.\n\nimpression\nFine."
    assert parse_answer(text, ASPECTS) == {"impression": "Fine.", "lungs_and_pleura": "Clear."}
    assert parse_answer("", ASPECTS) == {} and parse_answer("just one line", ASPECTS) == {}
