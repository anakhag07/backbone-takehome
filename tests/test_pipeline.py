"""End-to-end checks use a deterministic fake model, never a paid API call."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.pipeline import Settings, load_questions, run_pipeline
from src.storage import Store


TEXT = """Document ID: TEST-1
Example Behavioral Health
Patient: Alex Example | DOB: 1990-01-02 | MRN: EX-1
Encounter E-1 | Service date 2026-01-19
Actual patient contact: 09:00–09:40, 40 minutes.
"""


def extraction():
    return {
        "patients": [{"ref": "p1", "name": "Alex Example", "dob": "1990-01-02",
            "mrn": "EX-1", "issuer": "example behavioral health", "evidence": [TEXT.splitlines()[2]]}],
        "encounters": [{"patient_ref": "p1", "encounter_id": "E-1", "appointment_id": None,
            "service_date": "2026-01-19", "service_type": "individual", "record_kind": "clinical_note",
            "record_status": "final", "attendance": "attended", "patient_present": True,
            "contact_intervals": [{"start": "09:00", "end": "09:40"}], "presence_intervals": [],
            "therapy_intervals": [], "excluded_intervals": [], "reported_minutes": 40,
            "corrects_fields": [], "corrects_document_id": None, "is_copy": False,
            "evidence": TEXT.splitlines()[3:], "narrative": "Documented actual therapy.", "uncertainty": None}],
        "plans": [], "assessments": [], "observations": [], "warnings": [],
    }


class FakeModel:
    config_hash = "test-model-config-v1"

    def __init__(self, forbidden=False):
        self.calls = []
        self.forbidden = forbidden

    def called(self, kind):
        assert not self.forbidden, "Replay must not call the model"
        self.calls.append({"operation": kind, "usage": {"input_tokens": 1, "output_tokens": 1}})

    def extract_document(self, text, metadata):
        self.called("extract")
        return extraction()

    def plan_questions(self, questions, patients):
        self.called("plan")
        return {"scopes": [{"question_id": q["id"], "patient_ids": [patients[0]["patient_id"]],
            "start_date": "2026-01-19", "end_date": "2026-01-19", "focus_dates": [],
            "limitations": []} for q in questions]}

    def answer_question(self, question, context):
        self.called("answer")
        pid = context["scope"]["patient_ids"][0]
        overall = context["calculations"][pid]["period"]["overall"]
        assert overall["minutes"]["value"] == 40
        return {"answer": "One therapy encounter totaling 40 minutes.", "status": "answered",
            "source_doc_ids": context["allowed_source_doc_ids"],
            "calculation_keys": [f"calculations.{pid}.period.overall"], "limitations": []}


@pytest.fixture
def settings(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "note.txt").write_text(TEXT, encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "Q1", "question": "How much therapy did Alex receive?"}]))
    return Settings(questions, documents, tmp_path / "cache.sqlite3", tmp_path / "responses")


def read_result(result):
    return json.loads(result.answers_path.read_text())


def test_end_to_end_persists_and_replays_without_model(settings):
    first_model = FakeModel()
    first = run_pipeline(settings, first_model)
    assert first.status == "completed", first.error
    output = read_result(first)
    assert output["run"]["questions_source"]["path"] == str(settings.questions.resolve())
    assert len(output["run"]["questions_source"]["sha256"]) == 64
    assert output["run"]["timestamp_timezone"] == "UTC"
    assert first.answers_path.name == "answers.json"
    assert first.answers_path.parent.parent.name == output["run"]["started_at"][:10]
    assert len(first_model.calls) == 3
    second = run_pipeline(replace(settings, offline=True), FakeModel(forbidden=True))
    assert second.status == "completed", second.error
    assert second.answers_path != first.answers_path
    assert read_result(second)["answers"] == output["answers"]
    log = json.loads((second.answers_path.parent / "execution.json").read_text())
    assert log["cache"]["extractions_hit"] == 1
    assert log["cache"]["answers_hit"] == 1
    assert log["cache"]["abstractions_hit"] == 1
    assert log["model_call_count"] == 0
    with Store(settings.database) as store:
        patient = store.list_patients()[0]
        assert len(store.retrieve_documents(patient["patient_id"], "2026-01-19")) == 1


def test_exact_duplicate_does_not_change_answers_or_trigger_model(settings):
    first = run_pipeline(settings, FakeModel())
    assert first.status == "completed", first.error
    (settings.documents / "duplicate.txt").write_text(TEXT, encoding="utf-8")
    second = run_pipeline(replace(settings, offline=True), FakeModel(forbidden=True))
    assert second.status == "completed", second.error
    assert read_result(first)["answers"] == read_result(second)["answers"]
    with Store(settings.database) as store:
        assert len(store.list_documents()) == 1
        assert len(store.list_documents()[0]["receipts"]) == 2


def test_moved_corpus_reuses_work_and_exports_current_source_location(settings):
    first = run_pipeline(settings, FakeModel())
    assert first.status == "completed", first.error
    relocated = settings.documents.parent / "relocated"
    shutil.copytree(settings.documents, relocated)
    second = run_pipeline(replace(settings, documents=relocated, offline=True), FakeModel(forbidden=True))
    assert second.status == "completed", second.error
    assert read_result(first)["answers"] == read_result(second)["answers"]
    bundle = json.loads((second.answers_path.parent / "abstraction.json").read_text())
    assert next(iter(bundle["documents"].values()))["source_path"] == str(relocated / "note.txt")


def test_offline_miss_is_a_failed_timestamped_run(settings):
    result = run_pipeline(replace(settings, offline=True), FakeModel(forbidden=True))
    assert result.status == "failed"
    payload = read_result(result)
    assert "Offline cache miss" in payload["errors"][0]
    assert payload["answers"] == []
    assert (result.answers_path.parent / "execution.json").exists()


def test_changed_document_invalidates_extraction_cache(settings):
    first = run_pipeline(settings, FakeModel())
    assert first.status == "completed", first.error
    (settings.documents / "note.txt").write_text(TEXT + "Additional signed evidence.\n", encoding="utf-8")
    second = run_pipeline(replace(settings, offline=True), FakeModel(forbidden=True))
    assert second.status == "failed"
    assert "document note.txt" in second.error


def test_model_configuration_change_invalidates_cache(settings):
    assert run_pipeline(settings, FakeModel()).status == "completed"
    model = FakeModel(forbidden=True)
    model.config_hash = "different-model-config"
    result = run_pipeline(replace(settings, offline=True), model)
    assert result.status == "failed"
    assert "Offline cache miss" in result.error


def test_new_questions_reuse_extraction_and_materialized_abstraction(settings):
    assert run_pipeline(settings, FakeModel()).status == "completed"
    settings.questions.write_text('[{"id":"NEW","question":"A related new question"}]')

    class QuestionsOnly(FakeModel):
        def extract_document(self, text, metadata):
            raise AssertionError("New questions must reuse saved document extraction")

    client = QuestionsOnly()
    result = run_pipeline(settings, client)
    assert result.status == "completed", result.error
    assert [call["operation"] for call in client.calls] == ["plan", "answer"]
    execution = json.loads((result.answers_path.parent / "execution.json").read_text())
    assert execution["cache"]["abstractions_hit"] == 1


def test_focus_dates_do_not_mislabel_bounding_period_as_selected_total(settings):
    class FocusQuestions(FakeModel):
        def plan_questions(self, questions, patients):
            plan = super().plan_questions(questions, patients)
            plan["scopes"][0].update(end_date="2026-01-21", focus_dates=["2026-01-19", "2026-01-21"])
            return plan

        def answer_question(self, question, context):
            self.called("answer")
            result = next(iter(context["calculations"].values()))
            assert "period" not in result
            assert "bounding_period_context" in result
            assert set(result["focus_dates"]) == {"2026-01-19", "2026-01-21"}
            assert "intervening dates" in context["scope_note"]
            return {"answer": "Per-date results are available.", "status": "answered",
                "source_doc_ids": context["allowed_source_doc_ids"], "calculation_keys": [], "limitations": []}

    result = run_pipeline(settings, FocusQuestions())
    assert result.status == "completed", result.error


def test_failure_retains_completed_answers_and_cached_work(settings):
    settings.questions.write_text(json.dumps([
        {"id": "Q1", "question": "First question"}, {"id": "Q2", "question": "Fail now"}]))

    class PartialFailure(FakeModel):
        def answer_question(self, question, context):
            if question == "Fail now":
                raise RuntimeError("temporary model failure")
            return super().answer_question(question, context)

    first = run_pipeline(settings, PartialFailure())
    assert first.status == "failed"
    assert len(read_result(first)["answers"]) == 1
    second_model = FakeModel()
    second = run_pipeline(settings, second_model)
    assert second.status == "completed", second.error
    assert [call["operation"] for call in second_model.calls] == ["answer"]


def test_unverified_evidence_never_enters_extraction_cache(settings):
    class BadEvidence(FakeModel):
        def extract_document(self, text, metadata):
            value = super().extract_document(text, metadata)
            value["encounters"][0]["evidence"] = ["An invented quotation."]
            return value

    result = run_pipeline(settings, BadEvidence())
    assert result.status == "failed"
    with Store(settings.database) as store:
        document = store.list_documents()[0]
        assert store.get_extraction(document["doc_id"], FakeModel.config_hash) is None


def test_questions_validation_and_context_limit(settings):
    settings.questions.write_text('[{"id":"Q1","question":"A"},{"id":"Q1","question":"B"}]')
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(settings.questions)
    settings.questions.write_text('[{"id":"Q1","question":"A"}]')
    result = run_pipeline(replace(settings, max_context_chars=10), FakeModel())
    assert result.status == "failed"
    assert "No evidence was truncated" in result.error


def test_main_script_offline_failure_and_argument_help(settings):
    root = Path(__file__).resolve().parents[1]
    help_run = subprocess.run([sys.executable, str(root / "main.py"), "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0
    result = subprocess.run([sys.executable, str(root / "main.py"), str(settings.questions),
        "--offline", "--db", str(settings.database), "--response-dir", str(settings.responses)],
        capture_output=True, text=True)
    assert result.returncode == 1
    output_path = Path(result.stdout.strip())
    assert output_path.exists()
    assert json.loads(output_path.read_text())["run"]["status"] == "failed"
