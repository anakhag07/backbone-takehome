"""Generic notes, source requests, and citation checks without model access."""

from types import SimpleNamespace
import json

import pytest

import backbone.deferred_model as deferred
from backbone.model import ModelError


TEXT = "Record R1\nPatient attended from 09:00 to 09:40.\nDeparture corrected to 09:35.\n"


def notes():
    return {"summary": "An attendance account with a departure correction.", "facts": [
        {"text": "The original account reports a 09:40 departure.",
         "evidence": ["Patient attended from 09:00 to 09:40."]},
        {"text": "A departure correction reports 09:35.",
         "evidence": ["Departure corrected to 09:35."]},
    ], "warnings": []}


def answer(**changes):
    return {"answer": "A departure correction reports 09:35.", "status": "answered",
        "citations": [{"doc_id": "d1", "quote": "Departure corrected to 09:35."}],
        "limitations": [], "needs_source_doc_ids": [], **changes}


def context():
    return {"documents": [{"doc_id": "d1", "notes": deferred.validate_notes(notes(), TEXT)}]}


def mock_client(payload):
    captured = []
    responses = iter(payload) if isinstance(payload, list) else None

    def parse(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(status="completed", output_parsed=next(responses) if responses else payload,
            model="example-model", id="response-test", output=[], output_text="",
            usage={"input_tokens": 20, "output_tokens": 30})

    client = deferred.DeferredModelClient("example-model", "medium")
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    return client, captured


def test_notes_preserve_conflicting_accounts_and_regenerate_cached_locations():
    value = deferred.validate_notes(notes(), TEXT)
    assert len(value["facts"]) == 2
    quote = "Departure corrected to 09:35."
    span = value["source_spans"][quote][0]
    assert span["line_start"] == 3
    assert TEXT[span["char_start"]:span["char_end"]] == quote
    value["source_spans"][quote][0]["char_start"] = 0
    repaired = deferred.validate_notes(value, TEXT)
    assert repaired["source_spans"][quote][0]["char_start"] == TEXT.index(quote)


def test_fabricated_notes_are_not_accepted_for_cache():
    value = notes()
    value["facts"][0]["evidence"] = ["Patient departed at 10:00."]
    client, _ = mock_client(value)
    with pytest.raises(ModelError, match="source validation"):
        client.parse_document(TEXT, {})
    assert client.calls[-1]["validation_failed"] is True


def test_summary_limit_does_not_limit_fact_detail():
    value = notes()
    value["facts"][0]["text"] = "Detailed source interpretation. " * 100
    assert deferred.validate_notes(value, TEXT)["facts"] == value["facts"]
    value["summary"] = "x" * 2001
    with pytest.raises(ValueError):
        deferred.validate_notes(value, TEXT)


def test_collection_selection_cannot_silently_omit_documents():
    catalog = [{"doc_id": "d1"}, {"doc_id": "d2"}]
    value = {"scope_type": "collection_wide", "patient_labels": [],
        "doc_ids": ["d1"], "limitations": []}
    with pytest.raises(ValueError, match="every catalog"):
        deferred.validate_selection(value, catalog)
    value["doc_ids"].append("d2")
    assert deferred.validate_selection(value, catalog) == value


def test_selection_rejects_unknown_sources_and_ambiguous_individual_labels():
    catalog = [{"doc_id": "d1"}]
    value = {"scope_type": "individual_patient", "patient_labels": ["Alex"],
        "doc_ids": ["invented"], "limitations": []}
    with pytest.raises(ValueError, match="outside"):
        deferred.validate_selection(value, catalog)
    value.update(doc_ids=["d1"], patient_labels=["Alex", "Sam"])
    with pytest.raises(ValueError, match="exactly one"):
        deferred.validate_selection(value, catalog)


def test_original_request_then_answer_from_newly_visible_original_text():
    request = answer(answer="", status="unresolved", citations=[],
        limitations=["The notes do not establish the record identifier."], needs_source_doc_ids=["d1"])
    assert deferred.validate_answer(request, context())["needs_source_doc_ids"] == ["d1"]
    resolved = answer(answer="The record is R1.", citations=[{"doc_id": "d1", "quote": "Record R1"}])
    with pytest.raises(ValueError, match="does not occur"):
        deferred.validate_answer(resolved, context())
    full_context = context()
    full_context["documents"][0]["text"] = TEXT
    assert deferred.validate_answer(resolved, full_context) == resolved


def test_answer_cannot_cite_summary_paraphrases_or_join_separate_excerpts():
    value = answer(citations=[{"doc_id": "d1", "quote": notes()["summary"]}])
    with pytest.raises(ValueError, match="does not occur"):
        deferred.validate_answer(value, context())
    separated = {"documents": [{"doc_id": "d1", "notes": {"facts": [
        {"evidence": ["The duration was", "50 minutes."]}]}}]}
    value["citations"][0]["quote"] = "The duration was 50 minutes."
    with pytest.raises(ValueError, match="does not occur"):
        deferred.validate_answer(value, separated)


@pytest.mark.parametrize("field", ["citations", "needs_source_doc_ids"])
def test_answer_rejects_out_of_scope_document_references(field):
    value = answer(status="partial")
    value[field] = ([{"doc_id": "outside", "quote": "anything"}] if field == "citations" else ["outside"])
    with pytest.raises(ValueError, match="outside"):
        deferred.validate_answer(value, context())


def test_answered_requires_citations_and_no_outstanding_source_requests():
    for changes in ({"citations": []}, {"needs_source_doc_ids": ["d1"]}, {"answer": ""}):
        with pytest.raises(ValueError):
            deferred.validate_answer(answer(**changes), context())


def test_generic_request_uses_typed_notes_and_existing_usage_reporting():
    client, captured = mock_client(notes())
    result = client.parse_document(TEXT, {"source_doc_id": "R1"})
    assert result["source_spans"]
    assert captured[0]["text_format"] is deferred.DocumentNotes
    assert captured[0]["reasoning"] == {"effort": "medium"}
    assert captured[0]["store"] is False
    assert client.calls[0]["operation"] == "parse_document"
    assert client.calls[0]["usage"]["input_tokens"] == 20


def test_answer_validation_failure_is_recorded_in_telemetry():
    value = answer(citations=[{"doc_id": "d1", "quote": "Unsupported quote"}])
    client, captured = mock_client(value)
    with pytest.raises(ModelError, match="source validation"):
        client.answer_from_notes("When did they leave?", context())
    assert client.calls[-1]["validation_failed"] is True
    assert len(captured) == len(client.calls) == 2
    assert all(call["failed"] is True for call in client.calls)
    assert all(call["usage"]["output_tokens"] == 30 for call in client.calls)


def test_invalid_citation_is_corrected_once_using_the_same_context():
    bad = answer(citations=[{"doc_id": "d1", "quote": "Departure was corrected to 09:35."}])
    good = answer()
    client, captured = mock_client([bad, good])
    supplied = context()
    assert client.answer_from_notes("When did they leave?", supplied) == good
    assert [call["operation"] for call in client.calls] == ["answer_from_notes", "answer_from_notes_repair"]
    assert client.calls[0]["failed"] is True
    assert not client.calls[1].get("failed", False)
    retried = json.loads(captured[1]["input"])
    assert retried["context"] == supplied
    assert retried["rejected_answer"] == bad
    assert "does not occur" in retried["validation_error"]


def test_valid_answer_does_not_make_a_correction_call():
    client, captured = mock_client(answer())
    assert client.answer_from_notes("When did they leave?", context()) == answer()
    assert len(captured) == len(client.calls) == 1


def test_answer_prompt_changes_preserve_document_note_cache_identity(monkeypatch):
    first = deferred.DeferredModelClient("example-model", "medium")
    monkeypatch.setattr(deferred, "ANSWER_INSTRUCTIONS", deferred.ANSWER_INSTRUCTIONS + " New answer wording.")
    second = deferred.DeferredModelClient("example-model", "medium")
    assert first.extraction_config_hash == second.extraction_config_hash
    assert first.routing_hash == second.routing_hash
    assert first.answer_hash != second.answer_hash
    assert first.config_hash != second.config_hash
    assert second.extraction_config_hash != deferred.DeferredModelClient("example-model", "high").extraction_config_hash


def test_generic_schemas_are_strict_required_objects():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "default" not in node
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    for model in (deferred.DocumentNotes, deferred.DocumentSelection, deferred.EvidenceAnswer):
        check(model.model_json_schema())
