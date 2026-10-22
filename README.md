# Backbone take-home

## 1. Setup and running

Python 3.11+, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
export OPENAI_API_KEY="your-api-key"

python main.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json" \
  --strategy eager --db .backbone/abstraction.sqlite3
```

Add or update UTF-8 `.txt` files in the document directory and rerun. Keep the complete active collection there. To ask new questions using the saved abstraction:

```bash
python main.py /path/to/new_questions.json \
  --documents "../Backbone_ Building Clinical Intelligence 09.27/documents" \
  --strategy eager --db .backbone/abstraction.sqlite3
```

Question format: `[{"id": "Q1", "question": "..."}]`, with unique IDs. Reuse the same database across restarts. `--offline` requires cached answers; `--refresh` recomputes model work. This experiment branch defaults to `lazy`, so pass `--strategy eager` as shown. Run checks with `python -m pytest -q`.

To use the alternate implementation on `experiment/deferred-document-reading`, first index documents without making model calls (no API key needed), then answer questions:

```bash
python main.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json" \
  --strategy lazy --ingest-only
python main.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json" \
  --strategy lazy
```

`lazy` uses `.backbone/deferred.sqlite3`; `eager` uses `.backbone/abstraction.sqlite3`. Storage never resets automatically, including across restarts or Git branch changes. A new `--db` path starts empty. `--refresh` bypasses caches and replaces applicable results; it does not erase historical documents. Changed content gets a new document identity, and changed extraction instructions/model/effort get a new cache identity. Only files in the current document directory participate in answers.

## 2. Clinical abstraction and results

Logical schema, persisted in SQLite; extraction fields are defined in [models.py](backbone/models.py):

```text
documents[doc_id] -> original text, content hash, source metadata
extractions[(doc_id, configuration_hash)] -> patients, encounters,
    plans, assessments, observations, source_spans, warnings
patient_index[patient_id][service_date][encounter_id] -> {doc_ids}
patient_abstraction[patient_id] -> events, plans, assessments,
    observations, metrics, warnings
```

These are logical hash-map lookups; SQLite persists them with primary keys and indexes. Patient identity scopes the evidence, then service date brings accounts of the same day's care together for reconciliation. One date may contain several encounters. Document IDs retain separate originals, including corrections and copies. Cheap header extraction uses explicit labels and regular expressions: it is fast and inspectable, but neither a header match nor a date mention proves a clinical event occurred.

The alternate `lazy` path stores `document_notes[(doc_id, configuration_hash)] -> {summary, facts with quotes, warnings}`. Intake only stores text and deterministic header/date candidates. At question time, the model selects a patient; the code includes all records sharing the scoped MRN plus records with unknown identity. Selected documents are parsed once into generic notes, without therapy-specific fields or precomputed totals. The summary is limited to 2,000 characters; evidence-linked facts are not. If notes are insufficient, the answer model can request originals for one reread pass. Exact questions reuse saved answers. Original rereads affect that answer cache; they do not currently enrich the reusable notes.

For a fixed corpus and extraction configuration, eager parses all `N` documents; lazy parses only the union `U` of documents selected by questions. The potential saving is `N - |U|` document readings. In this supplied single-patient corpus, the first question selects all 31 documents, so `|U| = N`: deferral shifts the work into question time without avoiding those readings.

The abstraction retains conflicting alternatives and supporting quotations; the source registry includes their locations. Python reconciles encounters and calculates session counts, durations, weekly totals, and plan compliance.

Each run saves `answers.json`, `abstraction.json`, `execution.json`, and `execution-summary.json` under [responses/](responses/). The final submission run will be identified later. See the [benchmark](docs/benchmark.md) for the intake/question time split, model usage, cost, and saved abstraction size.

Run directories now include a full UTC timestamp, for example `responses/2026-10-22/2026-10-22T11-14-21.349286Z-4c80de08/answers.json`. `run.answer_id` is `answers-<run-id>`; the short random suffix prevents collisions, rather than hashing the answer. Individual answers retain their input question IDs.

`execution.json` records individual events. `execution-summary.json` reports initial processing per document, completed question latency by scope, model-call latency, usage/cost estimates, saved abstraction size, and each question's percentage of active/selected documents already parsed **before** it starts. A question is one input item; model calls are SDK requests for extraction, routing, answering, rereading or citation repair. A question can trigger many calls, or zero on an exact cache hit. Failed attempts remain visible and are excluded from completed-answer latency averages. Missing measurements are `null`.

## 3. Approach and tradeoffs

**Decision:** eager prioritizes repeated questions about a stable clinical schema: the model extracts claims and explains results, while Python calculates numerical answers. Lazy offers generic notes and shifts parsing to the first question that needs each document. Eager intake took **178.68 s** for 31 documents; lazy text-only intake took **0.061 s**. On the same two uncached questions with all documents already parsed, eager averaged **15.27 s** and lazy **22.75 s**. Lazy's earlier cold attempt spent **219.97 s** parsing before failing citation validation. Citation handling was repaired, but subsequent HTTP 429 responses prevented completing the remaining lazy questions. Successful cold lazy latency and collection-wide lazy latency remain unmeasured. See [benchmark results](benchmark-results.json) and the [measurement notes](docs/benchmark.md).

**Limitations:** intake errors can persist in the cached abstraction; there is no targeted correction-and-invalidation workflow. The schema favors the supplied therapy, plan, and progress questions, so new question types may require new fields and re-extraction. Next, test deliberately incorrect extractions and held-out question types, then add versioned corrections and recompute affected patient records.

**At 500K+ documents:** both `run_pipeline` in [pipeline.py](backbone/pipeline.py) and `run_deferred_pipeline` in [deferred.py](backbone/deferred.py) scan/hash the complete directory and load the active corpus into memory on every run. Use a change manifest and load text by indexed document ID only when needed. Both extraction loops make serial API calls; use bounded workers with rate-limit handling. Lazy `_answer_questions` sends the full catalog for routing, then all selected notes for answering; the 400,000-character guard will reject large contexts. Replace catalog-wide model routing with indexed patient lookup, and use scoped/sharded aggregation with explicit coverage for collection totals. Top-k retrieval alone cannot establish complete counts. SQLite's [single-writer constraint](https://www.sqlite.org/whentouse.html) also matters as concurrent ingestion grows. These are code-based scaling expectations; this scale has not been measured.

**Models and assistance:** `gpt-5.6-terra`, reasoning effort `medium`; configurable with `--model` and `--effort`. Default context guard: 400,000 characters. Codex assisted with planning and coding; Claude assisted with PR reviews. The five supplied question queries averaged **about $0.29 each** in estimated model cost, excluding intake and shared scope planning; see the [benchmark assumptions](docs/benchmark.md).
