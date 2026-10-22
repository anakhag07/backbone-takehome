# Backbone take-home

Turn clinical documents into a reusable patient record, then answer a JSON question file. LLM extracts clinical claims and writes explanations; Python reconciles encounters and calculates counts, durations, and weekly totals.

## Storage design

Two logical stores separate patient-level evidence from original document text:

```text
patient_index[patient_id][service_date][encounter_id] -> {doc_ids}
patient_abstraction[patient_id] -> encounters, plans, assessments, conflicts

document_store[doc_id] -> immutable text, SHA-256, source ID/path, receipts
extractions[(doc_id, configuration_hash)] -> structured claims, source spans
```

This is a hash-map-style lookup model persisted in **SQLite**, a local database file requiring no server. Physically, patient/identifier, abstraction, document-reference, document/receipt, and extraction/cache tables implement these stores. SQLite uses B-tree indexes: `(patient_id, service_date, encounter_id, doc_id)` supports patient/date lookup, and the document primary key supports retrieval by `doc_id`.

`patient_id` is an internal UUID. An issuing system's MRN links records to that ID; name and DOB provide supporting identity evidence. Names alone are not unique. `doc_id` is a separate UUID for an immutable document revision. The content hash detects identical text and reuses extraction; changed text creates a new revision. Text is stored once even when referenced by several dates or patients.

**Why patient, then date?** Define `I(p, d)` as the documents with evidence about patient `p` on service date `d`. This isolates a patient's record, brings competing accounts of the same care day together, and supports chronological and weekly calculations. Multiple encounters can occur that day, so date alone never merges encounters. A document can appear under several dates; a later correction links to the date of care it describes. Unknown dates retain patient-level references. Patient-level plans remain available across dates. The index supports focused retrieval; the current question path still supplies the selected patient's full abstraction.

## Deterministic extraction

Explicit header labels are cheap and repeatable to parse with rules. The current parser extracts document ID, source institution, and patient name/DOB/MRN candidates, recording its method separately from model extraction. Python also handles reconciliation and arithmetic so totals follow inspectable rules.

Clinical narratives, corrections, and eligibility require interpretation: **each uncached document currently receives one structured LLM extraction**, even when its header matches. Claims retain quotations checked against the source, and unresolved conflicts remain alternatives. Cached extraction avoids repeat calls for unchanged content and configuration. Template-specific clinical rules with selective LLM fallback are a future optimization; matching a header does not establish full document coverage.

## Setup

Use Python 3.11+ and run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
export OPENAI_API_KEY="your-api-key"
```

### 1. Process new documents

Put UTF-8 `.txt` files in a document directory; subdirectories are included. Ingestion and answering happen in the same command:

```bash
python main.py /path/to/questions.json --documents /path/to/documents
```

Add new or updated files to that directory and rerun. Unchanged documents reuse cached extraction. Keep the complete active collection there: removing a file removes it from the current evidence set. The default database is `.backbone/abstraction.sqlite3` in the working directory. Use a separate `--db /path/to/other.sqlite3` for a different collection.

### 2. Ask new questions

Create a JSON array with a unique, nonempty string `id` and a nonempty string `question` for each item:

```json
[
  {"id": "Q1", "question": "How many therapy minutes did Rowan receive on January 19, 2026?"}
]
```

An object shaped as `{"questions": [...]}` is also accepted. Reuse the same document directory and database when asking a new file:

```bash
python main.py /path/to/new_questions.json --documents /path/to/documents
```

Without `--documents`, the pipeline uses `documents/` beside the questions file. For the supplied local materials:

```bash
python main.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json"
```

Model defaults are `gpt-5.6-terra` and `medium`; override with `--model MODEL --effort EFFORT` or `OPENAI_MODEL` / `OPENAI_REASONING_EFFORT`. Effort must be supported by the selected model. `--offline` replays fully cached runs and fails on missing extraction, question-scope, or answer caches; new questions normally need API calls. `--refresh` recomputes model work. See `python main.py --help` for all options.

## Output

Each run prints the absolute answer path; progress goes to stderr. Output is grouped by UTC date and time:

```text
responses/YYYY-MM-DD/HHMMSS.microsecondsZ-unique-id/
  answers.json      # Answers, calculated results, source question path/hash
  abstraction.json  # Patient records, source IDs, quotations and locations
  execution.json    # Runtime, cache activity, model calls and token usage
```

Answers retain question IDs, model settings, uncertainty, and provisional document/calculation references. Citation presentation will be refined next. Failed runs return a nonzero exit code and retain partial output marked `failed`. Change the output directory with `--response-dir`.

## Validation and limitations

Run `python -m pytest -q`. Tests use synthetic records and mocked model calls; live API accuracy has not been validated. Quote matching establishes that evidence exists, not that its interpretation is correct. Calculated results come from Python; numerical consistency of model prose is not independently checked. Supported weekly calculations use Monday–Sunday weeks and patient-present individual/group/family therapy; unsupported rules and unresolved evidence are flagged.

Codex assisted with design, implementation, and tests. Dependencies are pinned in `requirements.lock.txt`; saved extractions support downstream replay despite model variability.
