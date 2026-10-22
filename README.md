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

The abstraction retains conflicting alternatives and supporting quotations; the source registry includes their locations. Python reconciles encounters and calculates session counts, durations, weekly totals, and plan compliance.

Each run saves `answers.json`, `abstraction.json`, `execution.json`, and `execution-summary.json` under [responses/](responses/). The final submission run will be identified later. See the [benchmark](docs/benchmark.md) for the intake/question time split, model usage, cost, and saved abstraction size.

## 3. Approach and tradeoffs

**Decision:** prioritize lower per-question processing time by extracting and reconciling documents upfront, then reusing the clinical abstraction. The model extracts claims and explains results; Python calculates numerical answers. Measured eager intake took **178.68 s** for 31 documents; subsequent uncached patient questions averaged **12.72 s**. The alternative `lazy` strategy moved **219.97 s** of parsing into the first question, which failed citation validation. It has no completed-answer latency to compare.

**Limitations:** intake errors can persist in the cached abstraction; there is no targeted correction-and-invalidation workflow. The schema favors the supplied therapy, plan, and progress questions, so new question types may require new fields and re-extraction. Next, test deliberately incorrect extractions and held-out question types, then add versioned corrections and recompute affected patient records.

**At one million documents:** the first expected bottleneck is the full-directory scan and in-memory corpus in [pipeline.py](backbone/pipeline.py) and [deferred.py](backbone/deferred.py), followed by serial extraction and large question contexts. Use incremental intake, bounded workers, indexed patient retrieval, and per-patient aggregation. SQLite has no million-row ceiling; its [single-writer constraint](https://www.sqlite.org/whentouse.html) matters as concurrent ingestion grows. This scale has not been measured.

**Models and assistance:** `gpt-5.6-terra`, reasoning effort `medium`; configurable with `--model` and `--effort`. Default context guard: 400,000 characters. Codex assisted with planning and coding; Claude assisted with PR reviews. The five supplied question queries averaged **about $0.29 each** in estimated model cost, excluding intake and shared scope planning; see the [benchmark assumptions](docs/benchmark.md).
