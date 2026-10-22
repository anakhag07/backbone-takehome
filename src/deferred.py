"""Text-only intake; parse reusable generic notes when a question needs a document."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
import time

from .deferred_model import DeferredModelClient, _reference_context, validate_answer, validate_notes, validate_selection
from .model import _source_locations, parse_metadata
from .pipeline import RunResult, _now, _online, _safe_error, _write_json, digest, load_questions
from .reporting import make_run_id, model_call_stage, write_execution_summary
from .storage import Store

VERSION = "deferred-v1"


def _catalog_entry(document, text):
    metadata = parse_metadata(text)
    patient = metadata["patient"]
    key = None
    if patient and patient.get("issuer") and patient.get("mrn"):
        key = digest({"issuer": patient["issuer"], "mrn": patient["mrn"]})
    return {"doc_id": document["doc_id"], "source_document_id": document["source_doc_id"],
            "source_path": document["source_path"], "patient_key": key,
            "metadata": metadata, "date_mentions": sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)))}


def _expand_selection(selection, catalog):
    """Dates/titles never exclude plans or corrections; unknown headers stay candidates."""
    if selection["scope_type"] == "collection_wide":
        return sorted(item["doc_id"] for item in catalog)
    if selection["scope_type"] == "unresolved":
        return []
    chosen = set(selection["doc_ids"])
    keys = {item["patient_key"] for item in catalog if item["doc_id"] in chosen and item["patient_key"]}
    for item in catalog:
        if not item["patient_key"] or item["patient_key"] in keys:
            chosen.add(item["doc_id"])
    return sorted(chosen)


def _visible_notes(notes):
    # Character offsets belong in saved evidence; the answer model only needs quotations.
    return {key: value for key, value in notes.items() if key != "source_spans"}


def _check_context(settings, context):
    size = len(json.dumps(context, ensure_ascii=False))
    if size > settings.max_context_chars:
        raise ValueError(f"Question context has {size} characters; increase --max-context-chars. No evidence was truncated.")
    return size


def _readable_citations(answer, citations):
    """Render model-invented private citation markup as numbered source references."""
    numbers = {f"{item['doc_id']}:e-{hashlib.sha256(item['quote'].encode()).hexdigest()[:16]}": str(i)
               for i, item in enumerate(citations, 1)}

    def replace_marker(match):
        matched = list(dict.fromkeys(numbers[ref] for ref in match.group(1).split("\ue202") if ref in numbers))
        return f"[{', '.join(matched)}]" if matched else ""

    return re.sub(r"\ue200cite\ue202([^\ue201]*)\ue201", replace_marker, answer)


def run_deferred_pipeline(settings, client=None, progress=None, *, ingest_only=False):
    report = progress or (lambda message: None)
    started = datetime.now(timezone.utc)
    run_id = make_run_id(started)
    folder = settings.responses.resolve() / started.date().isoformat() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    answers_path = folder / "answers.json"
    clock = time.perf_counter()
    client = client or DeferredModelClient(settings.model, settings.effort)
    payload = {"schema_version": 1, "run": {
        "id": run_id, "answer_id": f"answers-{run_id}", "strategy": "lazy", "pipeline_version": VERSION,
        "started_at": started.isoformat(), "finished_at": None, "timestamp_timezone": "UTC", "status": "running",
        "questions_source": {"path": str(settings.questions.resolve()), "sha256": None},
        "documents_directory": str(settings.documents.resolve()), "database_path": str(settings.database.resolve()),
        "model": {"provider": "openai", "name": settings.model, "reasoning_effort": settings.effort,
                  "configuration_hash": client.config_hash}, "offline": settings.offline, "ingest_only": ingest_only},
        "answers": [], "errors": [], "artifacts": {"abstraction": "abstraction.json", "execution": "execution.json",
                                                    "execution_summary": "execution-summary.json"}}
    execution = {"strategy": "lazy", "telemetry_version": 1, "documents": [], "questions": [], "model_calls": [],
                 "cache": {"extractions_hit": 0, "extractions_miss": 0, "plans_hit": 0,
                           "plans_miss": 0, "answers_hit": 0, "answers_miss": 0}}
    documents, notes, catalog = {}, {}, []
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
        files = sorted(p for p in settings.documents.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
        if not files:
            raise ValueError("No UTF-8 .txt documents found in the documents directory.")
        intake_clock = time.perf_counter()
        with Store(settings.database) as store:
            report(f"Text index and note cache: {settings.database.resolve()}")
            for index, path in enumerate(files, 1):
                doc_clock = time.perf_counter()
                text = path.read_text(encoding="utf-8-sig")
                metadata = parse_metadata(text)
                document = store.ingest_document(path, text, metadata["source_doc_id"], metadata["source_system"])
                doc_id = document["doc_id"]
                duplicate = doc_id in documents
                if not duplicate:
                    document["source_path"] = str(path.resolve())
                    documents[doc_id] = document
                    entry = _catalog_entry(document, text)
                    catalog.append(entry)
                    store.save_cache("deferred_index", doc_id, entry)
                    cached = None if settings.refresh else store.get_extraction(doc_id, client.extraction_config_hash)
                    if cached is not None:
                        notes[doc_id] = validate_notes(cached, text)
                execution["documents"].append({"doc_id": doc_id, "path": str(path.resolve()),
                    "status": "duplicate" if duplicate else "indexed", "model_call": False,
                    "elapsed_seconds": time.perf_counter() - doc_clock})
                report(f"Document {index}/{len(files)}: {path.name} [text indexed; no model call]")
            execution["ingestion_seconds"] = time.perf_counter() - intake_clock
            execution["initial_processing_seconds"] = time.perf_counter() - intake_clock
            execution["active_document_count"] = len(documents)
            execution["parsed_documents_after_intake"] = len(notes)
            if not ingest_only:
                _answer_questions(settings, client, store, questions, catalog, documents, notes,
                                  execution, payload, answers_path, report)
        payload["run"]["status"] = "completed"
    except Exception as error:
        payload["run"]["status"] = "failed"
        payload["errors"].append(_safe_error(error))
    finally:
        payload["run"]["finished_at"] = _now()
        execution["total_seconds"] = time.perf_counter() - clock
        execution["model_calls"] = list(client.calls)
        execution["model_call_count"] = len(client.calls)
        execution["api_cost_usd"] = None
        execution["cost_note"] = "No API invoice available; usage retained for separately labeled pricing estimates."
        execution["database_bytes"] = settings.database.stat().st_size if settings.database.exists() else 0
        _write_json(folder / "abstraction.json", {"schema_version": 1, "strategy": "lazy", "catalog": catalog,
                    "document_notes": notes, "parsed_document_count": len(notes), "active_document_count": len(documents)})
        _write_json(folder / "execution.json", execution)
        _write_json(answers_path, payload)
        write_execution_summary(folder, execution, payload["run"], payload["answers"])
    return RunResult(answers_path, payload["run"]["status"], payload["errors"][0] if payload["errors"] else None)


def _answer_questions(settings, client, store, questions, catalog, documents, notes,
                      execution, payload, answers_path, report):
    # Source locations can change without changing the answer/evidence identity.
    routing_catalog = [{key: value for key, value in entry.items() if key != "source_path"} for entry in catalog]
    for question in questions:
        qclock, call_start = time.perf_counter(), len(client.calls)
        measurement = {"question_id": question["id"], "status": "failed", "cache_hit": False,
            "scope_type": "unresolved", "active_document_count": len(documents),
            "parsed_documents_before": len(notes),
            "documents_already_parsed_percent": 100 * len(notes) / len(documents)}
        execution["questions"].append(measurement)
        report(f"Answering {question['id']} [{len(notes)}/{len(documents)} documents already parsed]")
        try:
            routing_question = json.dumps({"current_question": question, "question_set_for_references": questions}, ensure_ascii=False)
            routing_key = digest({"question": routing_question, "catalog": routing_catalog, "config": client.routing_hash})
            selection = None if settings.refresh else store.get_cache("deferred_routes", routing_key)
            if selection is None:
                _online(settings, f"document selection for {question['id']}")
                _check_context(settings, {"question": routing_question, "catalog": routing_catalog})
                with model_call_stage(client, "document_selection", question_id=question["id"]):
                    selection = validate_selection(client.select_documents(routing_question, routing_catalog), routing_catalog)
                store.save_cache("deferred_routes", routing_key, selection)
                execution["cache"]["plans_miss"] += 1
            else:
                selection = validate_selection(selection, routing_catalog)
                execution["cache"]["plans_hit"] += 1
            selected = _expand_selection(selection, catalog)
            selection = {**selection, "doc_ids": selected}
            selected_before = sum(doc_id in notes for doc_id in selected)
            measurement.update(scope_type=selection["scope_type"], selected_document_count=len(selected),
                selected_documents_parsed_before=selected_before,
                selected_documents_already_parsed_percent=100 * selected_before / len(selected) if selected else None)
            parse_clock = time.perf_counter()
            for doc_id in selected:
                if doc_id in notes:
                    execution["cache"]["extractions_hit"] += 1
                    continue
                _online(settings, f"document notes {doc_id}")
                report(f"  Reading {documents[doc_id]['source_doc_id'] or doc_id} [model parse]")
                execution["cache"]["extractions_miss"] += 1
                metadata = next(entry for entry in catalog if entry["doc_id"] == doc_id)
                with model_call_stage(client, "deferred_document_parse", question_id=question["id"], doc_id=doc_id):
                    parsed = validate_notes(client.parse_document(documents[doc_id]["text"], metadata), documents[doc_id]["text"])
                store.save_extraction(doc_id, client.extraction_config_hash, parsed)
                notes[doc_id] = parsed
            measurement["deferred_parsing_seconds"] = time.perf_counter() - parse_clock
            measurement["documents_newly_parsed"] = len(selected) - selected_before
            context = {"question_set": questions, "selection": selection,
                "documents": [{"doc_id": doc_id, "source_document_id": documents[doc_id]["source_doc_id"],
                               "notes": _visible_notes(notes[doc_id])} for doc_id in selected]}
            measurement["context_characters"] = _check_context(settings, _reference_context(context)[0])
            answer_key = digest({"question": question, "context": context, "config": client.answer_hash, "version": VERSION})
            answer = None if settings.refresh else store.get_cache("deferred_answers", answer_key)
            if answer is None:
                execution["cache"]["answers_miss"] += 1
                if not selected:
                    answer = {"answer": "The requested patient could not be identified from the text index.", "status": "unresolved",
                              "citations": [], "limitations": selection["limitations"], "needs_source_doc_ids": []}
                else:
                    _online(settings, f"answer {question['id']}")
                    with model_call_stage(client, "question_answer", question_id=question["id"]):
                        answer = validate_answer(client.answer_from_notes(question["question"], context), context)
                    requested = set(answer["needs_source_doc_ids"])
                    if requested:
                        expanded = deepcopy(context)
                        for item in expanded["documents"]:
                            if item["doc_id"] in requested:
                                item["text"] = documents[item["doc_id"]]["text"]
                        measurement["context_characters"] = _check_context(settings, _reference_context(expanded)[0])
                        measurement["original_documents_reread"] = len(requested)
                        with model_call_stage(client, "source_reread_answer", question_id=question["id"]):
                            answer = validate_answer(client.answer_from_notes(question["question"], expanded), expanded)
                    if answer["needs_source_doc_ids"]:
                        answer["status"] = "partial"
                        answer["limitations"].append("One source reread pass was exhausted; additional evidence remains needed.")
                store.save_cache("deferred_answers", answer_key, answer)
            else:
                measurement["cache_hit"] = True
                execution["cache"]["answers_hit"] += 1
            # Always validate final citations against immutable originals, also on cache reuse.
            original_context = {"documents": [{"doc_id": doc_id, "text": documents[doc_id]["text"]} for doc_id in selected]}
            answer = validate_answer(answer, original_context)
            citations = [{**citation, "citation_number": index, "source_path": documents[citation["doc_id"]]["source_path"],
                          "source_document_id": documents[citation["doc_id"]]["source_doc_id"],
                          "locations": _source_locations(documents[citation["doc_id"]]["text"], citation["quote"])}
                         for index, citation in enumerate(answer["citations"], 1)]
            payload["answers"].append({"question_id": question["id"], "question": question["question"],
                "scope": selection, **answer, "answer": _readable_citations(answer["answer"], citations), "citations": citations})
            measurement["status"] = "completed"
            _write_json(answers_path, payload)
        finally:
            measurement["elapsed_seconds"] = time.perf_counter() - qclock
            measurement["parsed_documents_after"] = len(notes)
            calls = client.calls[call_start:]
            for call in calls:
                call["question_id"] = question["id"]
                call.setdefault("stage", "question")
            measurement["model_call_count"] = len(calls)
            measurement["model_seconds"] = sum(call.get("elapsed_seconds", 0) for call in calls)
