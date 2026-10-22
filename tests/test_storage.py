"""Behavioral tests for document reuse, identity boundaries, and patient indexes."""

import sqlite3

import pytest

from backbone.storage import Store


def identity(**changes):
    return {"name": "Example Patient", "dob": "1990-04-12", "mrn": "M-001", "issuer": "Clinic A", **changes}


def test_duplicate_text_reuses_document_and_extraction_after_restart(tmp_path):
    database = tmp_path / "saved.sqlite3"
    with Store(database) as store:
        first = store.ingest_document(tmp_path / "original.txt", "same source text", "D-1", "Clinic A")
        store.save_extraction(first["doc_id"], "config-v1", {"claims": [{"duration": 40}]})
        duplicate = store.ingest_document(tmp_path / "copy.txt", "same source text", "D-2", "Clinic A")
        assert first["doc_id"] == duplicate["doc_id"]
        assert len(duplicate["receipts"]) == 2
        assert {receipt["source_doc_id"] for receipt in duplicate["receipts"]} == {"D-1", "D-2"}
        assert len(store.list_documents()) == 1
    with Store(database) as store:
        assert store.get_document(first["doc_id"])["text"] == "same source text"
        assert store.get_extraction(first["doc_id"], "config-v1") == {"claims": [{"duration": 40}]}
        assert store.get_extraction(first["doc_id"], "config-v2") is None
        again = store.ingest_document(tmp_path / "copy.txt", "same source text", "D-2", "Clinic A")
        assert len(again["receipts"]) == 2


def test_changed_content_keeps_original_revision_and_cache(tmp_path):
    with Store(tmp_path / "saved.sqlite3") as store:
        original = store.ingest_document(tmp_path / "note.txt", "50 minutes", "D-1", "Clinic A")
        store.save_extraction(original["doc_id"], "cfg", {"minutes": 50})
        revision = store.ingest_document(tmp_path / "note.txt", "40 minutes", "D-1", "Clinic A")
        assert original["doc_id"] != revision["doc_id"]
        assert store.get_document(original["doc_id"])["text"] == "50 minutes"
        assert store.get_extraction(revision["doc_id"], "cfg") is None
        assert store.get_extraction(original["doc_id"], "cfg") == {"minutes": 50}


def test_patient_identity_uses_scoped_mrn_and_rejects_dob_conflicts(tmp_path):
    with Store(tmp_path / "saved.sqlite3") as store:
        first = store.resolve_patient(identity())
        assert store.resolve_patient(identity(name="Example Corrected", issuer=" clinic a ")) == first
        different_issuer = store.resolve_patient(identity(issuer="Clinic B"))
        assert different_issuer != first
        with pytest.raises(ValueError, match="Conflicting DOB"):
            store.resolve_patient(identity(dob="2000-01-01"))
        assert len(store.list_patients()) == 2
        matching = next(patient for patient in store.list_patients() if patient["patient_id"] == first)
        assert len(matching["identity_evidence"]) == 2
        assert matching["dob"] == "1990-04-12"


def test_missing_mrn_requires_unique_name_and_dob(tmp_path):
    with Store(tmp_path / "saved.sqlite3") as store:
        first = store.resolve_patient(identity())
        assert store.resolve_patient(identity(mrn=None, issuer=None, name=" example  patient ")) == first
        other = store.resolve_patient(identity(issuer="Clinic B"))
        assert other != first
        ambiguous = store.resolve_patient(identity(mrn=None, issuer=None))
        assert ambiguous not in {first, other}
        same_name_only = store.resolve_patient(identity(mrn=None, issuer=None, dob=None))
        assert same_name_only not in {first, other, ambiguous}


def test_date_index_allows_multiple_encounters_and_multiday_documents(tmp_path):
    with Store(tmp_path / "saved.sqlite3") as store:
        patient = store.resolve_patient(identity())
        register = store.ingest_document(tmp_path / "register.txt", "Several encounters", "D-1", "Clinic A")
        correction = store.ingest_document(tmp_path / "correction.txt", "Correct earlier departure", "D-2", "Clinic A")
        undated = store.ingest_document(tmp_path / "unknown.txt", "Unknown date", None, None)
        store.replace_references(patient, [
            {"service_date": "2026-01-19", "encounter_id": "E-1", "doc_id": register["doc_id"]},
            {"service_date": "2026-01-19", "encounter_id": "E-2", "doc_id": register["doc_id"]},
            {"service_date": "2026-01-21", "encounter_id": "E-3", "doc_id": register["doc_id"]},
            {"service_date": "2026-01-19", "encounter_id": "E-1", "doc_id": correction["doc_id"]},
            {"service_date": None, "encounter_id": None, "doc_id": undated["doc_id"]},
            {"service_date": None, "encounter_id": None, "doc_id": undated["doc_id"]},
        ])
        assert {row["doc_id"] for row in store.retrieve_documents(patient, "2026-01-19")} == {
            register["doc_id"], correction["doc_id"]
        }
        assert [row["doc_id"] for row in store.retrieve_documents(patient, "2026-01-21")] == [register["doc_id"]]
        assert len(store.retrieve_documents(patient)) == 3
        assert store.retrieve_documents(patient, "2026-01-26") == []
        with pytest.raises(sqlite3.IntegrityError):
            store.replace_references(patient, [{"service_date": None, "doc_id": "does-not-exist"}])
        assert len(store.retrieve_documents(patient)) == 3  # failed replacement rolled back


def test_abstraction_and_namespaced_cache_survive_restart(tmp_path):
    database = tmp_path / "saved.sqlite3"
    with Store(database) as store:
        patient = store.resolve_patient(identity())
        store.save_patient_abstraction(patient, {"events": [], "unresolved": ["duration"]})
        store.save_cache("question-plan", "same-key", {"kind": "count"})
        store.save_cache("answer", "same-key", {"count": 1})
    with Store(database) as store:
        assert store.resolve_patient(identity()) == patient
        assert store.get_patient_abstraction(patient) == {"events": [], "unresolved": ["duration"]}
        assert store.get_cache("question-plan", "same-key") == {"kind": "count"}
        assert store.get_cache("answer", "same-key") == {"count": 1}
        assert store.get_cache("answer", "missing") is None


def test_refuses_unknown_schema_version(tmp_path):
    database = tmp_path / "saved.sqlite3"
    with Store(database) as store:
        with store.connection:
            store.connection.execute("UPDATE schema_metadata SET value = '999' WHERE key = 'schema_version'")
    with pytest.raises(ValueError, match="Unsupported database schema"):
        Store(database)
