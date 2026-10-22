"""Small, explicit performance summaries for current and historical run bundles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import uuid


SCOPE_TYPES = ("individual_patient", "collection_wide", "unresolved")
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-terra"
PRICING_AS_OF = "2026-10-22"


def make_run_id(started: datetime | None = None) -> str:
    stamp = (started or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{stamp.strftime('%Y-%m-%dT%H-%M-%S.%fZ')}-{uuid.uuid4().hex[:8]}"


def classify_scope(scope: dict, question: str = "") -> str:
    explicit = scope.get("scope_type")
    if explicit in SCOPE_TYPES:
        return explicit
    patients = scope.get("patient_ids", [])
    if not patients:
        return "unresolved"
    if len(patients) > 1 or re.search(
        r"\b(?:collection[- ]wide|cohort|all patients|each patient|every patient|across patients)\b",
        question, re.IGNORECASE,
    ):
        return "collection_wide"
    return "individual_patient"


@contextmanager
def model_call_stage(client, stage: str, question_id: str | None = None,
                     doc_id: str | None = None):
    """Tag calls, including failed requests, without changing request/cache identity."""
    first = len(client.calls)
    try:
        yield
    finally:
        for call in client.calls[first:]:
            call.update(stage=stage, question_id=question_id)
            if doc_id is not None:
                call["doc_id"] = doc_id


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values):
    samples = [value for value in values if _number(value)]
    return statistics.fmean(samples) if samples else None


def _percentage(part, whole):
    return 100 * part / whole if _number(part) and _number(whole) and whole > 0 else None


def _latencies(rows):
    completed = [row for row in rows if row.get("status", "completed") == "completed"]
    measured = [row["elapsed_seconds"] for row in completed if _number(row.get("elapsed_seconds"))]
    return {
        "sample_count": len(rows), "completed_count": len(completed),
        "failed_count": sum(row.get("status") == "failed" for row in rows),
        "timed_completed_count": len(measured),
        "average_latency_seconds": _mean(measured),
        "total_latency_seconds": sum(measured) if measured else None,
        "average_cached_latency_seconds": _mean(row.get("elapsed_seconds") for row in completed if row.get("cache_hit") is True),
        "average_uncached_latency_seconds": _mean(row.get("elapsed_seconds") for row in completed if row.get("cache_hit") is False),
    }


def _usage(calls):
    fields = {"input_tokens": ("input_tokens",), "cached_input_tokens": ("input_tokens_details", "cached_tokens"),
              "output_tokens": ("output_tokens",), "reasoning_tokens": ("output_tokens_details", "reasoning_tokens"),
              "total_tokens": ("total_tokens",)}
    result = {"calls_with_usage": sum(isinstance(call.get("usage"), dict) for call in calls),
              "call_count": len(calls)}
    for name, path in fields.items():
        values = []
        for call in calls:
            value = call.get("usage")
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if _number(value):
                values.append(value)
        result[name] = sum(values) if values else (0 if not calls else None)
        result[f"{name}_reported_call_count"] = len(values)
    return result


def _estimated_cost(calls):
    """Price reported token usage, never substitute an estimate for a billed cost."""
    priced = []
    for call in calls:
        usage = call.get("usage")
        if call.get("requested_model") != "gpt-5.6-terra" or not isinstance(usage, dict):
            continue
        details = usage.get("input_tokens_details") or {}
        inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
        cached, written = details.get("cached_tokens"), details.get("cache_write_tokens", 0)
        if not all(_number(value) and value >= 0 for value in (inputs, outputs, cached, written)):
            continue
        if cached + written > inputs:
            continue
        # Published long-context multipliers apply above 272K input tokens.
        input_multiplier = 2 if inputs > 272_000 else 1
        output_multiplier = 1.5 if inputs > 272_000 else 1
        cost = ((inputs - cached - written) * 2 * input_multiplier
                + cached * 0.20 * input_multiplier
                + written * 2 * input_multiplier * 1.25
                + outputs * 12 * output_multiplier) / 1_000_000
        priced.append(cost)
    return {
        "estimated_cost_usd": sum(priced) if len(priced) == len(calls) else None,
        "known_partial_estimated_cost_usd": sum(priced) if priced else (0 if not calls else None),
        "priced_call_count": len(priced), "unpriced_call_count": len(calls) - len(priced),
        "pricing_source": PRICING_SOURCE, "pricing_as_of": PRICING_AS_OF,
        "estimate_note": "Token-based list-price estimate for gpt-5.6-terra, not an API invoice. Unknown usage/model rates leave the full estimate null; any known subtotal is reported separately.",
    }


def summarize_execution(execution: dict, run_metadata: dict, answers: list[dict] | None = None,
                        abstraction_bytes: int | None = None) -> dict:
    """Missing historical measurements stay null; averages never invent zero samples."""
    documents = execution.get("documents", [])
    calls = execution.get("model_calls", [])
    answers_by_id = {answer["question_id"]: answer for answer in answers or []}
    unique_ids = {item["doc_id"] for item in documents if item.get("doc_id")}
    unique_count = execution.get("active_document_count", len(unique_ids))
    statuses = {item.get("status") for item in documents if item.get("status") != "duplicate"}
    cache_state = ("text_only" if statuses and statuses <= {"indexed", "text_indexed"} else
                   "cold" if statuses == {"extracted"} else "warm" if statuses == {"cached"} else
                   "mixed" if statuses else "unavailable")
    initial = execution.get("initial_processing_seconds", execution.get("ingestion_seconds"))
    questions = []
    for original in execution.get("questions", []):
        row = dict(original)
        answer = answers_by_id.get(row.get("question_id"), {})
        row.setdefault("scope_type", classify_scope(answer.get("scope", {}), answer.get("question", "")))
        row.setdefault("status", "completed" if answer else "unknown")
        # Older eager runs completed extraction before every recorded question.
        if "parsed_documents_before" not in row and documents and statuses <= {"cached", "extracted"}:
            row.update(parsed_documents_before=len(unique_ids), active_document_count=unique_count,
                       coverage_source="inferred_from_completed_eager_intake")
        row["documents_already_parsed_percent"] = _percentage(row.get("parsed_documents_before"), row.get("active_document_count"))
        row["selected_documents_already_parsed_percent"] = _percentage(
            row.get("selected_documents_parsed_before"), row.get("selected_document_count"))
        related = [call for call in calls if call.get("question_id") == row.get("question_id")]
        row["attributed_model_call_count"] = len(related)
        row["attributed_model_call_seconds"] = sum(call["elapsed_seconds"] for call in related if _number(call.get("elapsed_seconds"))) if related else (0 if execution.get("telemetry_version") else None)
        questions.append(row)
    extracted = [item for item in documents if item.get("status") == "extracted"]
    return {
        "schema_version": 1, "run_id": run_metadata.get("id"),
        "status": run_metadata.get("status"), "strategy": run_metadata.get("strategy", "eager"),
        "model": run_metadata.get("model"), "measurement_kind": "observed",
        "total_seconds": execution.get("total_seconds"),
        "initial_processing": {
            "total_seconds": initial,
            "timing_scope": ("text_index_only" if run_metadata.get("strategy") == "lazy" else "document_intake_and_initial_abstraction") if "initial_processing_seconds" in execution else "document_intake_only_legacy_measurement",
            "document_scan_seconds": execution.get("ingestion_seconds"),
            "file_count": len(documents), "unique_document_count": unique_count,
            "average_seconds_per_document": initial / unique_count if _number(initial) and unique_count else None,
            "cache_state": cache_state,
            "new_extraction_count": len(extracted),
            "average_new_document_extraction_seconds": _mean(item.get("elapsed_seconds") for item in extracted),
            "cached_document_count": sum(item.get("status") == "cached" for item in documents),
        },
        "scope_planning_seconds": execution.get("scope_planning_seconds"),
        "question_latencies": {kind: _latencies([row for row in questions if row.get("scope_type") == kind]) for kind in SCOPE_TYPES},
        "questions": questions,
        "model_calls": {
            "count": len(calls), "failed_count": sum(bool(call.get("failed") or call.get("validation_failed")) for call in calls),
            "timed_count": sum(_number(call.get("elapsed_seconds")) for call in calls),
            "average_latency_seconds": _mean(call.get("elapsed_seconds") for call in calls),
            "average_successful_latency_seconds": _mean(call.get("elapsed_seconds") for call in calls if not call.get("failed") and not call.get("validation_failed")),
            "total_latency_seconds": sum(call["elapsed_seconds"] for call in calls if _number(call.get("elapsed_seconds"))) if any(_number(call.get("elapsed_seconds")) for call in calls) else (0 if not calls else None),
            "by_stage": {stage: {"count": len(items), "average_latency_seconds": _mean(item.get("elapsed_seconds") for item in items)}
                         for stage in sorted({call.get("stage", "unattributed") for call in calls})
                         for items in [[call for call in calls if call.get("stage", "unattributed") == stage]]},
            "usage": _usage(calls), "api_cost_usd": execution.get("api_cost_usd"),
            **_estimated_cost(calls),
            "cost_note": execution.get("cost_note", "Pricing is unavailable; no cost is inferred from latency or token counts alone."),
        },
        "saved_abstraction_bytes": abstraction_bytes,
        "database_bytes": execution.get("database_bytes"),
        "notes": [
            "Question latency is application wall time for that question; model-call latency measures SDK requests, including any internal retries.",
            "Shared scope planning is reported separately when measured and is not silently allocated to individual questions.",
            "Document coverage is the fraction of active unique documents with saved parsing at question start; selected coverage uses that question's selected documents.",
            "Missing measurements and empty sample averages are null. Usage totals sum reported values; per-field reported-call counts show incomplete telemetry.",
            "Timings describe this run's cache state and workload, not a statistical benchmark or a 500K-document measurement.",
        ],
    }


def write_execution_summary(folder: Path, execution: dict, run_metadata: dict,
                            answers: list[dict] | None = None) -> dict:
    folder = Path(folder)
    abstraction = folder / "abstraction.json"
    summary = summarize_execution(execution, run_metadata, answers,
                                  abstraction.stat().st_size if abstraction.exists() else None)
    path = folder / "execution-summary.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create execution-summary.json for existing run directories without model calls.")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for folder in args.runs:
        execution = json.loads((folder / "execution.json").read_text())
        payload = json.loads((folder / "answers.json").read_text())
        write_execution_summary(folder, execution, payload["run"], payload.get("answers", []))
        print((folder / "execution-summary.json").resolve())


if __name__ == "__main__":
    main()
