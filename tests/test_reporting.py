"""Reporting checks keep cache states, missing measurements, and failures distinct."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from src.reporting import (classify_scope, make_run_id, model_call_stage,
                                summarize_execution, write_execution_summary)


def test_averages_separate_completed_scope_groups_from_failures_and_cache_hits():
    execution = {"telemetry_version": 2, "initial_processing_seconds": 20,
        "ingestion_seconds": 18, "active_document_count": 2,
        "documents": [{"doc_id": "a", "status": "extracted", "elapsed_seconds": 8},
                      {"doc_id": "b", "status": "extracted", "elapsed_seconds": 10}],
        "questions": [
            {"question_id": "one", "scope_type": "individual_patient", "status": "completed", "elapsed_seconds": 10, "cache_hit": False, "parsed_documents_before": 1, "active_document_count": 2, "selected_documents_parsed_before": 1, "selected_document_count": 1},
            {"question_id": "two", "scope_type": "individual_patient", "status": "completed", "elapsed_seconds": 2, "cache_hit": True},
            {"question_id": "failed", "scope_type": "individual_patient", "status": "failed", "elapsed_seconds": 100},
            {"question_id": "all", "scope_type": "collection_wide", "status": "completed", "elapsed_seconds": 30}],
        "model_calls": [{"question_id": "one", "stage": "question_answer", "elapsed_seconds": 8,
                         "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 20}, "output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 110}},
                        {"question_id": "failed", "stage": "question_answer", "elapsed_seconds": 90, "failed": True}]}
    summary = summarize_execution(execution, {"status": "failed", "strategy": "eager"})
    group = summary["question_latencies"]["individual_patient"]
    assert group["sample_count"] == 3 and group["failed_count"] == 1
    assert group["average_latency_seconds"] == 6
    assert group["average_uncached_latency_seconds"] == 10
    assert group["average_cached_latency_seconds"] == 2
    assert summary["question_latencies"]["collection_wide"]["average_latency_seconds"] == 30
    assert summary["initial_processing"]["average_seconds_per_document"] == 10
    assert summary["initial_processing"]["cache_state"] == "cold"
    assert summary["model_calls"]["average_latency_seconds"] == 49
    assert summary["model_calls"]["average_successful_latency_seconds"] == 8
    assert summary["model_calls"]["usage"]["input_tokens"] == 100
    assert summary["model_calls"]["usage"]["input_tokens_reported_call_count"] == 1
    assert summary["model_calls"]["usage"]["cached_input_tokens"] == 20
    assert summary["model_calls"]["api_cost_usd"] is None
    assert summary["questions"][0]["documents_already_parsed_percent"] == 50
    assert summary["questions"][0]["selected_documents_already_parsed_percent"] == 100


def test_missing_legacy_data_remains_unknown_and_empty_samples_are_not_zero():
    summary = summarize_execution({"model_calls": [{"operation": "old_call"}]}, {"status": "failed"})
    assert summary["initial_processing"]["total_seconds"] is None
    assert summary["model_calls"]["average_latency_seconds"] is None
    assert summary["model_calls"]["usage"]["input_tokens"] is None
    assert summary["question_latencies"]["collection_wide"]["average_latency_seconds"] is None
    assert summary["question_latencies"]["collection_wide"]["sample_count"] == 0


def test_backfill_infers_eager_coverage_and_scope_but_not_request_attribution(tmp_path):
    (tmp_path / "abstraction.json").write_text('{"patients": {}}\n')
    execution = {"ingestion_seconds": 4, "documents": [
        {"doc_id": "a", "status": "cached"}, {"doc_id": "a", "status": "duplicate"}],
        "questions": [{"question_id": "Q1", "elapsed_seconds": 2}], "model_calls": []}
    answers = [{"question_id": "Q1", "question": "How is this patient?", "scope": {"patient_ids": ["p1"]}}]
    summary = write_execution_summary(tmp_path, execution, {"id": "old", "status": "completed"}, answers)
    assert summary["initial_processing"]["unique_document_count"] == 1
    assert summary["initial_processing"]["timing_scope"] == "document_intake_only_legacy_measurement"
    assert summary["questions"][0]["documents_already_parsed_percent"] == 100
    assert summary["questions"][0]["attributed_model_call_seconds"] is None
    assert summary["question_latencies"]["individual_patient"]["average_latency_seconds"] == 2
    assert summary["saved_abstraction_bytes"] == (tmp_path / "abstraction.json").stat().st_size
    assert json.loads((tmp_path / "execution-summary.json").read_text()) == summary


def test_call_attribution_survives_failure_and_scope_handles_single_patient_collection():
    client = SimpleNamespace(calls=[])
    with pytest.raises(RuntimeError):
        with model_call_stage(client, "document_read", question_id="Q2", doc_id="d1"):
            client.calls.append({"failed": True})
            raise RuntimeError("request failed")
    assert client.calls[0] == {"failed": True, "stage": "document_read", "question_id": "Q2", "doc_id": "d1"}
    assert classify_scope({"patient_ids": ["p1"]}, "Review all patients in the collection") == "collection_wide"
    assert classify_scope({"patient_ids": ["p1"]}, "Review this patient's care") == "individual_patient"
    assert classify_scope({"patient_ids": []}) == "unresolved"
    first = make_run_id(datetime(2026, 10, 22, 3, 4, 5, tzinfo=timezone.utc))
    second = make_run_id(datetime(2026, 10, 22, 3, 4, 5, tzinfo=timezone.utc))
    assert first.startswith("2026-10-22T03-04-05.000000Z-") and first != second


def test_cost_estimates_use_reported_cache_usage_and_leave_unpriced_calls_unknown():
    call = {"requested_model": "gpt-5.6-terra", "usage": {"input_tokens": 1_000_000,
        "input_tokens_details": {"cached_tokens": 200_000, "cache_write_tokens": 100_000},
        "output_tokens": 10_000}}
    result = summarize_execution({"model_calls": [call]}, {})["model_calls"]
    # Long-context prices: 700K * $4/M + 200K * $.4/M + 100K * $5/M + 10K * $18/M.
    assert result["estimated_cost_usd"] == pytest.approx(3.56)
    assert result["api_cost_usd"] is None
    short = {"requested_model": "gpt-5.6-terra", "usage": {"input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 100}}
    result = summarize_execution({"model_calls": [short, {"failed": True}]}, {})["model_calls"]
    assert result["estimated_cost_usd"] is None
    assert result["known_partial_estimated_cost_usd"] == pytest.approx(0.00302)
    assert result["priced_call_count"] == result["unpriced_call_count"] == 1
