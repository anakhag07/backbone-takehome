"""Persistent document snapshots and patient/date indexes using SQLite.

Stored extraction payloads and patient abstractions are application data. This
module never promotes an extracted claim to a clinical conclusion.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid


SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise TypeError("Stored payloads must be dictionaries.")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Identity fields must be strings or null.")
    return value.strip() or None


def _name_key(value: str | None) -> str | None:
    return " ".join(value.split()).casefold() if value else None


class Store:
    """A local persistent store; public methods commit their own atomic writes."""

    def __init__(self, db_path: str | Path):
        database = str(db_path)
        if database != ":memory:":
            path = Path(database).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            database = str(path)
        self.db_path = database
        self.connection = sqlite3.connect(database, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = self.connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is not None and row["value"] != str(SCHEMA_VERSION):
            self.connection.close()
            raise ValueError(f"Unsupported database schema version: {row['value']}")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                source_path TEXT NOT NULL,
                source_doc_id TEXT,
                source_system TEXT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_external_id
                ON documents(source_system, source_doc_id);
            CREATE TABLE IF NOT EXISTS document_receipts (
                receipt_id INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                source_path TEXT NOT NULL,
                source_doc_id TEXT NOT NULL DEFAULT '',
                source_system TEXT NOT NULL DEFAULT '',
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(doc_id, source_path, source_doc_id, source_system)
            );
            CREATE TABLE IF NOT EXISTS extractions (
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                config_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(doc_id, config_hash)
            );
            CREATE TABLE IF NOT EXISTS caches (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(namespace, cache_key)
            );
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                name_key TEXT,
                dob TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS patients_demographics ON patients(name_key, dob);
            CREATE TABLE IF NOT EXISTS patient_identifiers (
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                issuer TEXT NOT NULL,
                issuer_key TEXT NOT NULL,
                mrn TEXT NOT NULL,
                UNIQUE(issuer_key, mrn)
            );
            CREATE INDEX IF NOT EXISTS identifiers_patient ON patient_identifiers(patient_id);
            CREATE TABLE IF NOT EXISTS identity_evidence (
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(patient_id, payload_json)
            );
            CREATE TABLE IF NOT EXISTS patient_abstractions (
                patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id),
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_references (
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                service_date TEXT,
                encounter_id TEXT,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS references_unique
                ON document_references(patient_id, COALESCE(service_date, ''),
                    COALESCE(encounter_id, ''), doc_id);
            CREATE INDEX IF NOT EXISTS references_patient_date
                ON document_references(patient_id, service_date, encounter_id, doc_id);
            CREATE INDEX IF NOT EXISTS references_document ON document_references(doc_id);
            """
        )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def __enter__(self) -> Store:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def _write(self) -> Iterator[None]:
        # Taking the write lock before lookup keeps check-and-insert operations
        # atomic when two local processes ingest the same source simultaneously.
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def ingest_document(
        self,
        path: Path,
        text: str,
        source_doc_id: str | None,
        source_system: str | None,
    ) -> dict[str, Any]:
        """Store one immutable text snapshot and record where it was received.

        Exact UTF-8 text duplicates share a document ID even if received from
        another path. A changed body is a new revision with a new internal ID.
        ``source_path`` on the canonical document is its first observed path;
        ``receipts`` preserves all observed paths and their source metadata.
        """
        source_path = str(Path(path).expanduser().resolve())
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now = _now()
        with self._write():
            row = self.connection.execute(
                "SELECT doc_id, text FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            if row is None:
                doc_id = str(uuid.uuid4())
                self.connection.execute(
                    """INSERT INTO documents
                    (doc_id, content_hash, source_path, source_doc_id, source_system, text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, content_hash, source_path, source_doc_id, source_system, text, now),
                )
            else:
                if row["text"] != text:
                    raise ValueError("Content-hash collision; refusing to merge source documents.")
                doc_id = row["doc_id"]
            self.connection.execute(
                """INSERT INTO document_receipts
                (doc_id, source_path, source_doc_id, source_system, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id, source_path, source_doc_id, source_system)
                DO UPDATE SET last_seen_at = excluded.last_seen_at""",
                (doc_id, source_path, source_doc_id or "", source_system or "", now, now),
            )
        return self.get_document(doc_id)

    def _document(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        receipts = self.connection.execute(
            """SELECT source_path, source_doc_id, source_system, first_seen_at, last_seen_at
            FROM document_receipts WHERE doc_id = ? ORDER BY receipt_id""",
            (row["doc_id"],),
        ).fetchall()
        result["receipts"] = [
            {**dict(receipt), "source_doc_id": receipt["source_doc_id"] or None,
             "source_system": receipt["source_system"] or None}
            for receipt in receipts
        ]
        return result

    def get_document(self, doc_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown document: {doc_id}")
        return self._document(row)

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM documents ORDER BY created_at, doc_id").fetchall()
        return [self._document(row) for row in rows]

    def get_extraction(self, doc_id: str, config_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM extractions WHERE doc_id = ? AND config_hash = ?",
            (doc_id, config_hash),
        ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_extraction(self, doc_id: str, config_hash: str, payload: dict[str, Any]) -> None:
        serialized = _json(payload)
        with self._write():
            self.connection.execute(
                """INSERT INTO extractions(doc_id, config_hash, payload_json, updated_at)
                VALUES (?, ?, ?, ?) ON CONFLICT(doc_id, config_hash)
                DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at""",
                (doc_id, config_hash, serialized, _now()),
            )

    def get_cache(self, namespace: str, key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM caches WHERE namespace = ? AND cache_key = ?", (namespace, key)
        ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def save_cache(self, namespace: str, key: str, payload: dict[str, Any]) -> None:
        serialized = _json(payload)
        with self._write():
            self.connection.execute(
                """INSERT INTO caches(namespace, cache_key, payload_json, updated_at)
                VALUES (?, ?, ?, ?) ON CONFLICT(namespace, cache_key)
                DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at""",
                (namespace, key, serialized, _now()),
            )

    def resolve_patient(self, identity: dict[str, Any]) -> str:
        """Resolve a scoped MRN or unambiguous exact name/DOB, preserving evidence.

        A new scoped MRN creates a new patient even when demographics happen to
        match an existing identified patient. Missing issuer makes an MRN
        unscoped, so it is retained as evidence without a global uniqueness claim.
        """
        name = _value(identity.get("name"))
        dob = _value(identity.get("dob"))
        mrn = _value(identity.get("mrn"))
        issuer = _value(identity.get("issuer"))
        if dob is not None:
            try:
                if date.fromisoformat(dob).isoformat() != dob:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("Patient DOB must be an ISO date (YYYY-MM-DD).") from exc
        normalized_name = _name_key(name)
        issuer_key = _name_key(issuer)
        evidence = _json(identity)
        with self._write():
            patient = None
            if issuer and mrn:
                patient = self.connection.execute(
                    """SELECT p.* FROM patients p JOIN patient_identifiers i
                    ON p.patient_id = i.patient_id WHERE i.issuer_key = ? AND i.mrn = ?""",
                    (issuer_key, mrn),
                ).fetchone()
                if patient and patient["dob"] and dob and patient["dob"] != dob:
                    raise ValueError("Conflicting DOB for an existing scoped MRN; identity linkage requires review.")
            elif normalized_name and dob:
                matches = self.connection.execute(
                    "SELECT * FROM patients WHERE name_key = ? AND dob = ?",
                    (normalized_name, dob),
                ).fetchall()
                if len(matches) == 1:
                    patient = matches[0]

            if patient is None:
                patient_id = str(uuid.uuid4())
                self.connection.execute(
                    "INSERT INTO patients(patient_id, name, name_key, dob, created_at) VALUES (?, ?, ?, ?, ?)",
                    (patient_id, name, normalized_name, dob, _now()),
                )
                if issuer and mrn:
                    self.connection.execute(
                        "INSERT INTO patient_identifiers(patient_id, issuer, issuer_key, mrn) VALUES (?, ?, ?, ?)",
                        (patient_id, issuer, issuer_key, mrn),
                    )
            else:
                patient_id = patient["patient_id"]
                # Fill absent canonical fields; preserve differing name spellings
                # in identity_evidence instead of silently replacing old evidence.
                self.connection.execute(
                    """UPDATE patients SET name = COALESCE(name, ?),
                    name_key = COALESCE(name_key, ?), dob = COALESCE(dob, ?) WHERE patient_id = ?""",
                    (name, normalized_name, dob, patient_id),
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO identity_evidence(patient_id, payload_json, created_at) VALUES (?, ?, ?)",
                (patient_id, evidence, _now()),
            )
        return patient_id

    def list_patients(self) -> list[dict[str, Any]]:
        patients = self.connection.execute("SELECT * FROM patients ORDER BY created_at, patient_id").fetchall()
        result = []
        for row in patients:
            patient = dict(row)
            patient.pop("name_key")
            patient["identifiers"] = [
                dict(identifier)
                for identifier in self.connection.execute(
                    "SELECT issuer, mrn FROM patient_identifiers WHERE patient_id = ? ORDER BY issuer_key, mrn",
                    (row["patient_id"],),
                )
            ]
            patient["identity_evidence"] = [
                json.loads(evidence["payload_json"])
                for evidence in self.connection.execute(
                    "SELECT payload_json FROM identity_evidence WHERE patient_id = ? ORDER BY created_at",
                    (row["patient_id"],),
                )
            ]
            result.append(patient)
        return result

    def save_patient_abstraction(self, patient_id: str, payload: dict[str, Any]) -> None:
        serialized = _json(payload)
        with self._write():
            self.connection.execute(
                """INSERT INTO patient_abstractions(patient_id, payload_json, updated_at)
                VALUES (?, ?, ?) ON CONFLICT(patient_id)
                DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at""",
                (patient_id, serialized, _now()),
            )

    def get_patient_abstraction(self, patient_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM patient_abstractions WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def replace_references(self, patient_id: str, references: list[dict[str, Any]]) -> None:
        rows = set()
        for reference in references:
            service_date = _value(reference.get("service_date"))
            if service_date is not None:
                try:
                    if date.fromisoformat(service_date).isoformat() != service_date:
                        raise ValueError
                except ValueError as exc:
                    raise ValueError("Service dates must be ISO dates (YYYY-MM-DD) or null.") from exc
            encounter_id = _value(reference.get("encounter_id"))
            doc_id = _value(reference.get("doc_id"))
            if not doc_id:
                raise ValueError("Each document reference requires doc_id.")
            rows.add((patient_id, service_date, encounter_id, doc_id))
        with self._write():
            if not self.connection.execute(
                "SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone():
                raise KeyError(f"Unknown patient: {patient_id}")
            self.connection.execute("DELETE FROM document_references WHERE patient_id = ?", (patient_id,))
            self.connection.executemany(
                "INSERT INTO document_references(patient_id, service_date, encounter_id, doc_id) VALUES (?, ?, ?, ?)",
                rows,
            )

    def retrieve_documents(self, patient_id: str, service_date: str | None = None) -> list[dict[str, Any]]:
        sql = """SELECT DISTINCT d.* FROM documents d JOIN document_references r
            ON d.doc_id = r.doc_id WHERE r.patient_id = ?"""
        params: list[str] = [patient_id]
        if service_date is not None:
            sql += " AND r.service_date = ?"
            params.append(service_date)
        sql += " ORDER BY d.created_at, d.doc_id"
        return [self._document(row) for row in self.connection.execute(sql, params).fetchall()]
