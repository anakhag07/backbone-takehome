"""Question-triggered parsing, conservative retrieval, and cache recovery offline."""

from dataclasses import replace
import hashlib
import json

import pytest

from src.deferred import run_deferred_pipeline
from src.pipeline import Settings
from src.storage import Store


def document(source_id, token, *, name="Alex Example", mrn="A-1"):
    identity = f"Patient: {name} | DOB: 1990-01-02 | MRN: {mrn}\n" if name else ""
    return f"Document ID: {source_id}\nExample Behavioral Health\n{identity}Observed token {token}.\n"


@pytest.fixture
def settings(tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()
    for filename, text in {
        "a1.txt": document("A1", "alpha"),
        "a2.txt": document("A2", "beta", name="Alex Example corrected"),
        "b1.txt": document("B1", "gamma", name="Sam Example", mrn="B-1"),
        "unknown.txt": document("U1", "delta", name=None),
    }.items():
        (directory / filename).write_text(text)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "Q1", "question": "Summarize Alex's record."}]))
    return Settings(questions, directory, tmp_path / "cache.sqlite3", tmp_path / "responses")


class FakeDeferredModel:
    config_hash = "fake-deferred-v1"
    extraction_config_hash = "fake-notes-v1"
    routing_hash = "fake-routing-v1"
    answer_hash = "fake-answer-v1"

    def __init__(self, forbidden=False):
        self.forbidden = forbidden
        self.calls = []
        self.parsed_sources = []
        self.answer_contexts = []

    def called(self, operation):
        assert not self.forbidden, "Cached replay must make no model calls"
        self.calls.append({"operation": operation, "elapsed_seconds": 0.01,
                           "usage": {"input_tokens": 1, "output_tokens": 1}})

    def select_documents(self, question, catalog):
        self.called("select_documents")
        chosen = next(item for item in catalog if item["source_document_id"] == "A1")
        return {"scope_type": "individual_patient", "patient_labels": ["Alex Example"],
                "doc_ids": [chosen["doc_id"]], "limitations": []}

    def parse_document(self, text, metadata):
        self.called("parse_document")
        self.parsed_sources.append(metadata["source_document_id"])
        quote = text.strip().splitlines()[-1]
        return {"summary": "A documented observation.", "facts": [{"text": quote, "evidence": [quote]}],
                "warnings": []}

    def answer_from_notes(self, question, context):
        self.called("answer_from_notes")
        self.answer_contexts.append(context)
        item = context["documents"][0]
        return {"answer": "The source documents contain observations.", "status": "answered",
                "citations": [{"doc_id": item["doc_id"], "quote": item["notes"]["facts"][0]["evidence"][0]}],
                "limitations": [], "needs_source_doc_ids": []}


def artifact(result, name):
    return json.loads(result.answers_path.with_name(name).read_text())


def test_text_only_intake_persists_documents_without_any_model_work(settings):
    result = run_deferred_pipeline(settings, FakeDeferredModel(forbidden=True), ingest_only=True)
    assert result.status == "completed", result.error
    abstraction = artifact(result, "abstraction.json")
    execution = artifact(result, "execution.json")
    assert abstraction["active_document_count"] == 4
    assert abstraction["document_notes"] == {}
    assert execution["model_call_count"] == 0
    assert execution["questions"] == []
    assert artifact(result, "answers.json")["answers"] == []
    assert result.answers_path.with_name("execution-summary.json").exists()
    with Store(settings.database) as store:
        assert len(store.list_documents()) == 4
        assert store.connection.execute("SELECT count(*) FROM extractions").fetchone()[0] == 0


def test_question_expands_patient_identifier_and_unknown_headers(settings):
    client = FakeDeferredModel()
    result = run_deferred_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert set(client.parsed_sources) == {"A1", "A2", "U1"}
    supplied = {item["source_document_id"] for item in client.answer_contexts[0]["documents"]}
    assert supplied == {"A1", "A2", "U1"}
    assert set(client.answer_contexts[0]["selection"]["doc_ids"]) == {
        item["doc_id"] for item in client.answer_contexts[0]["documents"]}
    metric = artifact(result, "execution.json")["questions"][0]
    assert metric["documents_already_parsed_percent"] == 0
    assert metric["selected_documents_already_parsed_percent"] == 0
    assert metric["documents_newly_parsed"] == 3
    assert metric["parsed_documents_after"] == 3
    for call in artifact(result, "execution.json")["model_calls"]:
        assert call["question_id"] == "Q1"
        assert call["stage"] == {"select_documents": "document_selection",
                                 "parse_document": "deferred_document_parse",
                                 "answer_from_notes": "question_answer"}[call["operation"]]


def test_new_question_reuses_notes_and_exact_offline_replay_uses_no_model(settings):
    first = run_deferred_pipeline(settings, FakeDeferredModel())
    assert first.status == "completed", first.error
    replay = run_deferred_pipeline(replace(settings, offline=True), FakeDeferredModel(forbidden=True))
    assert replay.status == "completed", replay.error
    assert artifact(replay, "answers.json")["answers"] == artifact(first, "answers.json")["answers"]
    assert artifact(replay, "execution.json")["model_call_count"] == 0

    settings.questions.write_text(json.dumps([{"id": "Q2", "question": "Describe Alex's observations."}]))
    client = FakeDeferredModel()
    result = run_deferred_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert [call["operation"] for call in client.calls] == ["select_documents", "answer_from_notes"]
    metric = artifact(result, "execution.json")["questions"][0]
    assert metric["documents_already_parsed_percent"] == 75
    assert metric["selected_documents_already_parsed_percent"] == 100
    assert metric["documents_newly_parsed"] == 0


def test_changed_document_creates_revision_and_only_reparses_changed_source(settings):
    assert run_deferred_pipeline(settings, FakeDeferredModel()).status == "completed"
    (settings.documents / "a2.txt").write_text(document("A2", "updated beta", name="Alex Example corrected"))
    client = FakeDeferredModel()
    result = run_deferred_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert client.parsed_sources == ["A2"]
    assert artifact(result, "execution.json")["questions"][0]["documents_newly_parsed"] == 1
    with Store(settings.database) as store:
        assert len(store.list_documents()) == 5


def test_original_source_reread_is_cited_and_cached_for_offline_replay(settings):
    class NeedsOriginal(FakeDeferredModel):
        def answer_from_notes(self, question, context):
            self.called("answer_from_notes")
            self.answer_contexts.append(context)
            item = context["documents"][0]
            if "text" not in item:
                return {"answer": "", "status": "unresolved", "citations": [],
                        "limitations": ["Need the original record identifier."],
                        "needs_source_doc_ids": [item["doc_id"]]}
            return {"answer": "The original record identifies its document.", "status": "answered",
                    "citations": [{"doc_id": item["doc_id"], "quote": item["text"].splitlines()[0]}],
                    "limitations": [], "needs_source_doc_ids": []}

    client = NeedsOriginal()
    first = run_deferred_pipeline(settings, client)
    assert first.status == "completed", first.error
    assert len(client.answer_contexts) == 2
    metric = artifact(first, "execution.json")["questions"][0]
    assert metric["original_documents_reread"] == 1
    citation = artifact(first, "answers.json")["answers"][0]["citations"][0]
    assert citation["locations"][0]["line_start"] == 1
    replay = run_deferred_pipeline(replace(settings, offline=True), FakeDeferredModel(forbidden=True))
    assert replay.status == "completed", replay.error
    assert artifact(replay, "answers.json")["answers"] == artifact(first, "answers.json")["answers"]


def test_source_reread_stops_after_one_retry_with_explicit_partial_status(settings):
    class NeedsMore(FakeDeferredModel):
        def answer_from_notes(self, question, context):
            self.called("answer_from_notes")
            self.answer_contexts.append(context)
            return {"answer": "The evidence remains incomplete.", "status": "unresolved", "citations": [],
                    "limitations": ["Unresolved source detail."],
                    "needs_source_doc_ids": [context["documents"][0]["doc_id"]]}

    client = NeedsMore()
    result = run_deferred_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert len(client.answer_contexts) == 2
    output = artifact(result, "answers.json")["answers"][0]
    assert output["status"] == "partial"
    assert any("reread pass was exhausted" in reason for reason in output["limitations"])


def test_answer_failure_keeps_parsed_notes_for_retry(settings):
    class FailsAnswer(FakeDeferredModel):
        def answer_from_notes(self, question, context):
            self.called("answer_from_notes")
            raise RuntimeError("temporary model failure")

    first = run_deferred_pipeline(settings, FailsAnswer())
    assert first.status == "failed"
    assert artifact(first, "abstraction.json")["parsed_document_count"] == 3
    metric = artifact(first, "execution.json")["questions"][0]
    assert metric["status"] == "failed"
    assert metric["model_call_count"] == 5
    client = FakeDeferredModel()
    result = run_deferred_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert [call["operation"] for call in client.calls] == ["answer_from_notes"]


def test_private_citation_markup_is_rendered_as_numbered_source_references(settings):
    class MarkedAnswer(FakeDeferredModel):
        def answer_from_notes(self, question, context):
            result = super().answer_from_notes(question, context)
            citation = result["citations"][0]
            key = hashlib.sha256(citation["quote"].encode()).hexdigest()[:16]
            result["answer"] = f"An observation. \ue200cite\ue202{citation['doc_id']}:e-{key}\ue201"
            return result

    result = run_deferred_pipeline(settings, MarkedAnswer())
    assert result.status == "completed", result.error
    answer = artifact(result, "answers.json")["answers"][0]
    assert answer["answer"] == "An observation. [1]"
    assert answer["citations"][0]["citation_number"] == 1


def test_removed_documents_stop_contributing_but_history_is_preserved(settings):
    assert run_deferred_pipeline(settings, FakeDeferredModel()).status == "completed"
    (settings.documents / "a2.txt").unlink()
    client = FakeDeferredModel()
    result = run_deferred_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert client.parsed_sources == []
    assert {item["source_document_id"] for item in client.answer_contexts[0]["documents"]} == {"A1", "U1"}
    abstraction = artifact(result, "abstraction.json")
    assert abstraction["active_document_count"] == 3
    assert abstraction["parsed_document_count"] == 2
    with Store(settings.database) as store:
        assert len(store.list_documents()) == 4
