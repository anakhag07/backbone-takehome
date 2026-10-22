"""Persistent ingestion, deterministic calculations, and timestamped answer runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable
import uuid

from . import clinical
from .clinical import build_abstraction, calculate_metrics
from .model import ModelClient, parse_metadata, validate_extraction
from .models import QuestionAnswer, QuestionPlan
from .reporting import classify_scope, make_run_id, model_call_stage, write_execution_summary
from .storage import Store


PIPELINE_VERSION = "pipeline-v1"
COLLECTIONS = ("encounters", "plans", "assessments", "observations")


@dataclass(frozen=True)
class Settings:
    questions: Path
    documents: Path
    database: Path
    responses: Path
    model: str = "gpt-5.6-terra"
    effort: str = "medium"
    offline: bool = False
    refresh: bool = False
    max_context_chars: int = 400_000


@dataclass(frozen=True)
class RunResult:
    answers_path: Path
    status: str
    error: str | None = None


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_questions(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("questions")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Questions JSON must be a nonempty array of {id, question} objects.")
    questions = []
    ids = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Every question must be an object with id and question strings.")
        qid, question = entry.get("id"), entry.get("question")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("Every question needs a nonempty string id.")
        if qid in ids:
            raise ValueError(f"Duplicate question id: {qid}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {qid} needs nonempty question text.")
        ids.add(qid)
        questions.append({"id": qid, "question": question})
    return questions, hashlib.sha256(raw).hexdigest()


def _online(settings: Settings, operation: str) -> None:
    if settings.offline:
        raise ValueError(f"Offline cache miss for {operation}; run once with API access first.")


def _safe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    key = os.getenv("OPENAI_API_KEY")
    return message.replace(key, "[redacted]") if key else message


def _patient_records(store: Store, documents: dict, extracted: dict):
    """Resolve identity once, retaining document-local links across restarts."""
    records_by_patient: dict[str, list[dict]] = {}
    references_by_patient: dict[str, list[dict]] = {}
    links = {doc_id: {} for doc_id in extracted}
    mentions = [(doc_id, mention) for doc_id, payload in extracted.items()
                for mention in payload["patients"]]
    # Resolve strong identities before demographic-only mentions, independent of
    # random internal IDs and input filesystem order.
    mentions.sort(key=lambda pair: (not bool(pair[1]["mrn"] and pair[1]["issuer"]),
                                    documents[pair[0]]["content_hash"], pair[1]["ref"]))
    for doc_id, mention in mentions:
        identity = {key: mention[key] for key in ("name", "dob", "mrn", "issuer")}
        if identity["issuer"]:
            identity["issuer"] = " ".join(identity["issuer"].split()).casefold()
        link_key = digest({"doc_id": doc_id, "ref": mention["ref"], "identity": identity})
        saved_link = store.get_cache("identity_links", link_key)
        if saved_link:
            patient_id = saved_link["patient_id"]
        else:
            patient_id = store.resolve_patient({**identity, "doc_id": doc_id,
                                                "evidence": mention["evidence"]})
            store.save_cache("identity_links", link_key, {"patient_id": patient_id})
        links[doc_id][mention["ref"]] = patient_id
        records_by_patient.setdefault(patient_id, [])
        references_by_patient.setdefault(patient_id, []).append(
            {"doc_id": doc_id, "service_date": None, "encounter_id": None})
    for doc_id in sorted(extracted):
        payload, document = extracted[doc_id], documents[doc_id]
        ref_to_patient = links[doc_id]
        for patient_id in sorted(set(ref_to_patient.values())):
            record = {"doc_id": doc_id, "source_doc_id": document["source_doc_id"],
                      "warnings": payload["warnings"], "source_spans": payload["source_spans"]}
            for collection in COLLECTIONS:
                record[collection] = []
                for item in payload[collection]:
                    if ref_to_patient[item["patient_ref"]] != patient_id:
                        continue
                    linked = {**deepcopy(item), "patient_id": patient_id, "doc_id": doc_id,
                              "source_doc_id": document["source_doc_id"]}
                    record[collection].append(linked)
                    service_date = (item.get("service_date") or item.get("completed_date")
                                    or item.get("date") or item.get("effective_start"))
                    references_by_patient[patient_id].append({
                        "doc_id": doc_id, "service_date": service_date,
                        "encounter_id": item.get("encounter_id"),
                    })
            records_by_patient[patient_id].append(record)
    # The active corpus is the supplied documents directory. Clear links for patients
    # no longer represented in that corpus, preserving historical documents/extractions.
    for patient in store.list_patients():
        store.replace_references(patient["patient_id"],
                                 references_by_patient.get(patient["patient_id"], []))
    return records_by_patient


def _calculation_paths(value, prefix="calculations") -> list[str]:
    paths = [prefix]
    if isinstance(value, dict):
        for key, child in value.items():
            paths.extend(_calculation_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_calculation_paths(child, f"{prefix}[{index}]"))
    return paths


def _context(scope: dict, abstractions: dict, sources: dict) -> dict:
    selected = []
    calculations = {}
    doc_ids = set()
    for patient_id in scope["patient_ids"]:
        abstraction = abstractions[patient_id]
        scoped = calculate_metrics(abstraction["events"], abstraction["plans"],
                                   scope["start_date"], scope["end_date"])
        period_key = "bounding_period_context" if scope["focus_dates"] else "period"
        calculations[patient_id] = {period_key: scoped, "focus_dates": {
            day: calculate_metrics(abstraction["events"], abstraction["plans"], day, day)
            for day in scope["focus_dates"]}}
        # Keep the patient's evidence available for history/plan interpretation. The
        # explicitly scoped calculations determine which numbers answer the question.
        clinical_context = {key: value for key, value in abstraction.items()
                            if key not in {"metrics", "patient", "events"}}
        clinical_context["patient"] = {key: abstraction["patient"].get(key)
                                       for key in ("patient_id", "name", "dob")}
        clinical_context["events"] = [
            {key: value for key, value in event.items() if key != "claims"}
            for event in abstraction["events"]
        ]
        selected.append(clinical_context)
        doc_ids.update(abstraction["document_ids"])
    return {
        "scope": scope, "patients": selected, "calculations": calculations,
        "scope_note": ("Answer each focus date using its own metrics. bounding_period_context includes "
                       "intervening dates and is not a total for the selected dates. No combined selected-date "
                       "total is provided." if scope["focus_dates"] else "Period metrics use the inclusive requested date range."),
        # Paths can change when a corpus is moved or an identical file is added.
        # Keep them in the run's source registry, outside model/cache identity.
        "documents": [{key: value for key, value in sources[doc_id].items() if key != "source_path"}
                      for doc_id in sorted(doc_ids)],
        "allowed_source_doc_ids": sorted(doc_ids),
        "allowed_calculation_keys": _calculation_paths(calculations),
    }


def _validate_plan(plan: dict, questions: list[dict], patients: list[dict]) -> dict:
    checked = QuestionPlan.model_validate(plan).model_dump(mode="json")
    scopes = checked["scopes"]
    if sorted(s["question_id"] for s in scopes) != sorted(q["id"] for q in questions):
        raise ValueError("Question plan must contain each requested question exactly once.")
    valid_ids = {p["patient_id"] for p in patients}
    if any(not set(s["patient_ids"]).issubset(valid_ids) for s in scopes):
        raise ValueError("Question plan refers to an unavailable patient.")
    return checked


def _validate_answer(answer: dict, context: dict) -> dict:
    checked = QuestionAnswer.model_validate(answer).model_dump(mode="json")
    if not set(checked["source_doc_ids"]).issubset(context["allowed_source_doc_ids"]):
        raise ValueError("Answer refers to an unavailable source document.")
    if not set(checked["calculation_keys"]).issubset(context["allowed_calculation_keys"]):
        raise ValueError("Answer refers to an unavailable calculation.")
    return checked


def run_pipeline(settings: Settings, client=None,
                 progress: Callable[[str], None] | None = None) -> RunResult:
    """Write a run bundle even on failure; successful cached work remains reusable."""
    report = progress or (lambda message: None)
    started = datetime.now(timezone.utc)
    run_id = make_run_id(started)
    folder = settings.responses.resolve() / started.date().isoformat() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    answers_path = folder / "answers.json"
    clock = time.perf_counter()
    client = client or ModelClient(settings.model, settings.effort)
    payload = {
        "schema_version": 1,
        "run": {"id": run_id, "answer_id": f"answers-{run_id}", "strategy": "eager",
                "started_at": started.isoformat(), "finished_at": None,
                "timestamp_timezone": "UTC", "status": "running",
                "questions_source": {"path": str(settings.questions.resolve()), "sha256": None},
                "documents_directory": str(settings.documents.resolve()),
                "database_path": str(settings.database.resolve()),
                "model": {"provider": "openai", "name": settings.model,
                          "reasoning_effort": settings.effort,
                          "configuration_hash": client.config_hash},
                "offline": settings.offline, "pipeline_version": PIPELINE_VERSION},
        "answers": [], "errors": [],
        "artifacts": {"abstraction": "abstraction.json", "execution": "execution.json",
                      "execution_summary": "execution-summary.json"},
    }
    execution = {"telemetry_version": 2, "documents": [], "questions": [], "cache": {
        "extractions_hit": 0, "extractions_miss": 0, "abstractions_hit": 0, "abstractions_miss": 0, "plans_hit": 0,
        "plans_miss": 0, "answers_hit": 0, "answers_miss": 0}, "model_calls": []}
    abstractions, sources = {}, {}
    current_question = None
    _write_json(answers_path, payload)
    try:
        if settings.offline and settings.refresh:
            raise ValueError("--offline and --refresh cannot be used together.")
        if settings.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive.")
        questions, source_hash = load_questions(settings.questions)
        payload["run"]["questions_source"]["sha256"] = source_hash
        if not settings.documents.is_dir():
            raise ValueError(f"Documents directory does not exist: {settings.documents}")
        files = sorted(path for path in settings.documents.rglob("*")
                       if path.is_file() and path.suffix.lower() == ".txt")
        if not files:
            raise ValueError("No UTF-8 .txt documents found in the documents directory.")
        extraction_clock = time.perf_counter()
        report(f"Extraction cache: {settings.database.resolve()}")
        with Store(settings.database) as store:
            documents, extractions = {}, {}
            for index, path in enumerate(files, 1):
                label = f"Document {index}/{len(files)}: {path.name}"
                document_clock = time.perf_counter()
                text = path.read_text(encoding="utf-8-sig")
                metadata = parse_metadata(text)
                document = store.ingest_document(path, text, metadata["source_doc_id"],
                                                 metadata["source_system"])
                doc_id = document["doc_id"]
                if doc_id in documents:
                    report(f"{label} [duplicate; reusing extraction]")
                    execution["documents"].append({"path": str(path.resolve()),
                        "doc_id": doc_id, "status": "duplicate", "model_call": False,
                        "elapsed_seconds": time.perf_counter() - document_clock})
                    continue
                documents[doc_id] = document
                cached = None if settings.refresh else store.get_extraction(doc_id, client.config_hash)
                if cached is None:
                    execution["cache"]["extractions_miss"] += 1
                    _online(settings, f"document {path.name}")
                    reason = "forced refresh" if settings.refresh else "cache miss"
                    report(f"{label} [extracting with model; {reason}]")
                    metadata.update({"doc_id": doc_id, "source_path": str(path.resolve())})
                    with model_call_stage(client, "document_intake", doc_id=doc_id):
                        extraction = validate_extraction(client.extract_document(text, metadata), text)
                    store.save_extraction(doc_id, client.config_hash, extraction)
                else:
                    execution["cache"]["extractions_hit"] += 1
                    report(f"{label} [cached; no model call]")
                    extraction = validate_extraction(cached, text)
                extractions[doc_id] = extraction
                sources[doc_id] = {"doc_id": doc_id, "source_document_id": document["source_doc_id"],
                    "source_path": str(path.resolve()), "content_hash": document["content_hash"],
                    "deterministic_metadata": parse_metadata(text),
                    "source_spans": extraction["source_spans"],
                    "extraction_warnings": extraction["warnings"]}
                execution["documents"].append({"path": str(path.resolve()), "doc_id": doc_id,
                    "status": "cached" if cached is not None else "extracted",
                    "elapsed_seconds": time.perf_counter() - document_clock})
            execution["ingestion_seconds"] = time.perf_counter() - extraction_clock
            execution["active_document_count"] = len(documents)
            report(f"Documents: {execution['cache']['extractions_hit']} cached, "
                   f"{execution['cache']['extractions_miss']} extracted with model")
            report("Reconciling patient records and calculating care totals")
            records = _patient_records(store, documents, extractions)
            patients = [p for p in store.list_patients() if p["patient_id"] in records]
            for patient in patients:
                pid = patient["patient_id"]
                fingerprint = digest({"patient": patient, "records": records[pid],
                    "configuration": client.config_hash,
                    "rules": getattr(clinical, "RULE_VERSION", "clinical-v1")})
                prior_revision = store.get_cache("abstraction_revisions", pid)
                abstraction = store.get_patient_abstraction(pid)
                if (not settings.refresh and abstraction is not None and prior_revision
                        and prior_revision.get("fingerprint") == fingerprint):
                    execution["cache"]["abstractions_hit"] += 1
                else:
                    execution["cache"]["abstractions_miss"] += 1
                    abstraction = build_abstraction(patient, records[pid])
                    abstraction["document_ids"] = sorted(r["doc_id"] for r in records[pid])
                    store.save_patient_abstraction(pid, abstraction)
                    store.save_cache("abstraction_revisions", pid, {"fingerprint": fingerprint})
                abstractions[pid] = abstraction
            if not patients:
                raise ValueError("The documents did not establish any patient records.")
            execution["initial_processing_seconds"] = time.perf_counter() - extraction_clock
            planning_clock = time.perf_counter()
            planning_patients = []
            for patient in patients:
                abstract = abstractions[patient["patient_id"]]
                dates = sorted(e["date"] for e in abstract["events"] if e.get("date"))
                planning_patients.append({"patient_id": patient["patient_id"], "name": patient["name"],
                    "dob": patient["dob"], "date_start": dates[0] if dates else None,
                    "date_end": dates[-1] if dates else None, "plans": abstract["plans"]})
            plan_key = digest({"questions": questions, "patients": planning_patients,
                               "configuration": client.config_hash, "pipeline": PIPELINE_VERSION})
            plan = None if settings.refresh else store.get_cache("question_plans", plan_key)
            if plan is None:
                execution["cache"]["plans_miss"] += 1
                _online(settings, "question scope planning")
                report("Interpreting question scopes")
                with model_call_stage(client, "scope_planning"):
                    plan = _validate_plan(client.plan_questions(questions, planning_patients),
                                          questions, planning_patients)
                store.save_cache("question_plans", plan_key, plan)
            else:
                execution["cache"]["plans_hit"] += 1
                plan = _validate_plan(plan, questions, planning_patients)
            execution["scope_planning_seconds"] = time.perf_counter() - planning_clock
            scopes = {scope["question_id"]: scope for scope in plan["scopes"]}
            for question in questions:
                report(f"Answering {question['id']}")
                question_clock = time.perf_counter()
                scope = scopes[question["id"]]
                selected_docs = {doc_id for pid in scope["patient_ids"]
                                 for doc_id in abstractions[pid]["document_ids"]}
                current_question = {"question_id": question["id"], "status": "running",
                    "scope_type": classify_scope(scope, question["question"]),
                    "parsed_documents_before": len(extractions), "active_document_count": len(documents),
                    "documents_already_parsed_percent": 100 * len(extractions) / len(documents),
                    "selected_documents_parsed_before": len(selected_docs),
                    "selected_document_count": len(selected_docs),
                    "selected_documents_already_parsed_percent": 100.0 if selected_docs else None}
                execution["questions"].append(current_question)
                context = _context(scope, abstractions, sources)
                context_size = len(json.dumps(context, ensure_ascii=False))
                current_question["context_characters"] = context_size
                if context_size > settings.max_context_chars:
                    raise ValueError(f"Question {question['id']} context has {context_size} characters; "
                                     "narrow its scope or increase --max-context-chars. No evidence was truncated.")
                answer_key = digest({"question": question, "context": context,
                                     "configuration": client.config_hash, "pipeline": PIPELINE_VERSION})
                answer = None if settings.refresh else store.get_cache("answers", answer_key)
                hit = answer is not None
                if answer is None:
                    execution["cache"]["answers_miss"] += 1
                    if not scope["patient_ids"]:
                        answer = {"answer": "The requested patient could not be identified unambiguously.",
                                  "status": "unresolved", "source_doc_ids": [], "calculation_keys": [],
                                  "limitations": scope["limitations"]}
                    else:
                        _online(settings, f"answer {question['id']}")
                        with model_call_stage(client, "question_answer", question_id=question["id"]):
                            answer = client.answer_question(question["question"], context)
                    answer = _validate_answer(answer, context)
                    store.save_cache("answers", answer_key, answer)
                else:
                    execution["cache"]["answers_hit"] += 1
                    answer = _validate_answer(answer, context)
                payload["answers"].append({"question_id": question["id"], "question": question["question"],
                    "scope": scope, **answer, "calculated_results": context["calculations"]})
                current_question.update(status="completed", cache_hit=hit,
                                        elapsed_seconds=time.perf_counter() - question_clock)
                current_question = None
                _write_json(answers_path, payload)
        payload["run"]["status"] = "completed"
    except Exception as error:
        if current_question is not None:
            current_question.update(status="failed", elapsed_seconds=time.perf_counter() - question_clock)
        payload["run"]["status"] = "failed"
        payload["errors"].append(_safe_error(error))
    finally:
        payload["run"]["finished_at"] = _now()
        execution["total_seconds"] = time.perf_counter() - clock
        execution["model_calls"] = list(client.calls)
        execution["model_call_count"] = len(client.calls)
        execution["api_cost_usd"] = None
        execution["cost_note"] = "Not calculated; token usage is retained when reported by the API."
        execution["database_bytes"] = (settings.database.stat().st_size if settings.database.exists() else 0)
        _write_json(folder / "abstraction.json", {"schema_version": 1, "patients": abstractions,
                    "documents": sources, "pipeline_version": PIPELINE_VERSION})
        _write_json(folder / "execution.json", execution)
        _write_json(answers_path, payload)
        write_execution_summary(folder, execution, payload["run"], payload["answers"])
    return RunResult(answers_path, payload["run"]["status"],
                     payload["errors"][0] if payload["errors"] else None)
