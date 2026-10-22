"""Exercise the API boundary without credentials, SDK installation, or paid requests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from backbone.model import ModelClient, ModelError, parse_metadata, validate_extraction
from backbone.models import DocumentExtraction, QuestionAnswer, QuestionPlan


TEXT = """Document ID: DOC-1
Example Behavioral Health | Individual psychotherapy
Patient: Alex Example | DOB: 1990-01-02 | MRN: EX-1
Encounter E-1 | Service date 2026-01-19
Actual patient contact: 09:00–09:40, 40 minutes.
"""


def extraction():
    return {
        "patients": [{
            "ref": "p1", "name": "Alex Example", "dob": "1990-01-02",
            "mrn": "EX-1", "issuer": "example behavioral health",
            "evidence": ["Patient: Alex Example | DOB: 1990-01-02 | MRN: EX-1"],
        }],
        "encounters": [{
            "patient_ref": "p1", "encounter_id": "E-1", "appointment_id": None,
            "service_date": "2026-01-19", "service_type": "individual",
            "record_kind": "clinical_note", "record_status": "final",
            "attendance": "attended", "patient_present": True,
            "contact_intervals": [{"start": "09:00", "end": "09:40"}],
            "presence_intervals": [], "therapy_intervals": [], "excluded_intervals": [],
            "reported_minutes": 40, "corrects_fields": [], "corrects_document_id": None,
            "is_copy": False, "evidence": [
                "Encounter E-1 | Service date 2026-01-19",
                "Actual patient contact: 09:00–09:40, 40 minutes.",
            ],
            "narrative": "Actual contact is documented.", "uncertainty": None,
        }],
        "plans": [], "assessments": [], "observations": [], "warnings": [],
    }


def mock_client(payload, status="completed"):
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="resp-test", model="example-model", status=status,
            output_parsed=payload, usage={"input_tokens": 10, "output_tokens": 3},
            output=[], output_text="",
        )

    client = ModelClient("example-model", "medium")
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    return client, calls


def test_metadata_is_scoped_and_not_semantic_coverage():
    parsed = parse_metadata(TEXT)
    assert parsed["source_doc_id"] == "DOC-1"
    assert parsed["patient"]["mrn"] == "EX-1"
    assert parsed["patient"]["name"] == "Alex Example"
    assert parsed["semantic_coverage"] is False
    assert parsed["source_system"] == parse_metadata(TEXT.upper())["source_system"]


def test_verified_quotes_have_original_locations_and_survive_cache_read():
    payload = extraction()
    payload["patients"][0]["evidence"] = ["Patient: Alex   Example | DOB: 1990-01-02 | MRN: EX-1"]
    result = validate_extraction(payload, TEXT)
    span = next(iter(result["source_spans"].values()))[0]
    assert span["line_start"] == span["line_end"] == 3
    assert TEXT[span["char_start"]:span["char_end"]] == span["quote"]
    assert "Alex Example" in span["quote"]
    assert validate_extraction(result, TEXT) == result


def test_fabricated_source_quote_is_rejected():
    payload = extraction()
    payload["encounters"][0]["evidence"] = ["Actual contact: 50 minutes."]
    with pytest.raises(ValueError, match="does not occur"):
        validate_extraction(payload, TEXT)


@pytest.mark.parametrize("field,value", [
    ("patient_ref", "someone-else"),
    ("service_date", "2026-02-30"),
    ("reported_minutes", -1),
    ("contact_intervals", [{"start": "25:00", "end": "26:00"}]),
    ("contact_intervals", [{"start": "09:40", "end": "09:00"}]),
])
def test_invalid_claims_are_rejected(field, value):
    payload = extraction()
    payload["encounters"][0][field] = value
    with pytest.raises(ValueError):
        validate_extraction(payload, TEXT)


def test_config_hash_changes_for_model_and_effort_and_requires_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = ModelClient("first", "low")
    assert a.config_hash == ModelClient("first", "low").config_hash
    assert a.config_hash != ModelClient("second", "low").config_hash
    assert a.config_hash != ModelClient("first", "high").config_hash
    with pytest.raises(ModelError, match="OPENAI_API_KEY"):
        a.extract_document(TEXT, {})


def test_extraction_wiring_and_telemetry_do_not_contain_credentials():
    client, requests = mock_client(DocumentExtraction.model_validate(extraction()))
    client._api_key = "secret-test-key"
    result = client.extract_document(TEXT, parse_metadata(TEXT))
    assert result["encounters"][0]["reported_minutes"] == 40
    assert requests[0]["reasoning"] == {"effort": "medium"}
    assert requests[0]["store"] is False
    assert requests[0]["text_format"] is DocumentExtraction
    assert client.calls[0]["usage"]["input_tokens"] == 10
    assert "secret-test-key" not in repr(client.calls)


@pytest.mark.parametrize("status,parsed", [("incomplete", extraction()), ("completed", None)])
def test_incomplete_or_refused_output_is_explicit(status, parsed):
    client, _ = mock_client(parsed, status=status)
    with pytest.raises(ModelError):
        client.extract_document(TEXT, {})
    assert client.calls[0]["failed"] is True


def test_answer_rejects_unknown_sources_and_calculations():
    base = {
        "answer": "One encounter.", "status": "answered", "source_doc_ids": ["d1"],
        "calculation_keys": ["totals.sessions"], "limitations": [],
    }
    context = {"allowed_source_doc_ids": ["d1"], "allowed_calculation_keys": ["totals.sessions"]}
    client, _ = mock_client(base)
    assert client.answer_question("How many?", context) == base
    for key, bad in (("source_doc_ids", ["forged"]), ("calculation_keys", ["invented"])):
        payload = deepcopy(base)
        payload[key] = bad
        client, _ = mock_client(payload)
        with pytest.raises(ModelError, match="outside"):
            client.answer_question("How many?", context)


def test_answer_context_is_checked_before_request():
    client, requests = mock_client(None)
    with pytest.raises(ModelError, match="enumerate"):
        client.answer_question("Question", {})
    assert not requests


def test_question_planning_preserves_ids_and_date_scope():
    payload = {"scopes": [{
        "question_id": "Q1", "patient_ids": ["p1"], "start_date": "2026-01-19",
        "end_date": "2026-01-21", "focus_dates": ["2026-01-19", "2026-01-21"],
        "limitations": [],
    }]}
    questions = [{"id": "Q1", "question": "What happened on Jan 19 and Jan 21?"}]
    patients = [{"patient_id": "p1", "name": "Alex Example"}]
    client, _ = mock_client(payload)
    assert client.plan_questions(questions, patients) == payload
    bad_patient = deepcopy(payload)
    bad_patient["scopes"][0]["patient_ids"] = ["not-supplied"]
    client, _ = mock_client(bad_patient)
    with pytest.raises(ModelError, match="outside"):
        client.plan_questions(questions, patients)
    missing_question = {"scopes": []}
    client, _ = mock_client(missing_question)
    with pytest.raises(ModelError, match="exactly"):
        client.plan_questions(questions, patients)


def test_api_schema_objects_are_closed_and_fully_required():
    # Regression guard for the documented strict subset: required nullable fields,
    # closed objects, and nested anyOf are supported; root anyOf/composition is not.
    unsupported = {"allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else"}

    def check(node):
        if isinstance(node, list):
            for item in node:
                check(item)
            return
        if not isinstance(node, dict):
            return
        assert not unsupported.intersection(node)
        assert "default" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for value in node.values():
            check(value)

    for model in (DocumentExtraction, QuestionAnswer, QuestionPlan):
        schema = model.model_json_schema()
        assert schema["type"] == "object" and "anyOf" not in schema
        check(schema)


def test_transport_error_preserves_safe_status_and_request_id_only():
    class BadRequestError(Exception):
        status_code = 400
        request_id = "req_example_123"

    def fail(**kwargs):
        raise BadRequestError("Authorization: Bearer secret-test-key; private request body")

    client = ModelClient("example-model", "medium")
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=fail))
    with pytest.raises(ModelError) as failure:
        client.extract_document(TEXT, {})
    message = str(failure.value)
    assert "HTTP 400" in message and "req_example_123" in message
    assert "structured-output schema" in message
    assert "secret-test-key" not in message
    assert "private request body" not in repr(client.calls)
    assert client.calls[0]["http_status"] == 400
    assert client.calls[0]["request_id"] == "req_example_123"
    assert failure.value.__suppress_context__ is True


def test_error_request_id_rejects_arbitrary_header_or_body_content():
    class AuthenticationError(Exception):
        status_code = 401
        request_id = "Authorization: Bearer secret-test-key"

    def fail(**kwargs):
        raise AuthenticationError("do not expose this")

    client = ModelClient("example-model", "medium")
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=fail))
    with pytest.raises(ModelError) as failure:
        client.extract_document(TEXT, {})
    assert "HTTP 401" in str(failure.value)
    assert "API key" in str(failure.value)
    assert "request_id" not in client.calls[0]
    assert "secret-test-key" not in str(failure.value)
