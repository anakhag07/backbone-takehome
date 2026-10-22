# Backbone take-home

A first runnable prototype for deriving persistent clinical records and answering a JSON question set. Documents are stored once, patient records reference documents by service date and encounter, and numerical answers are calculated in Python. OpenAI extracts semantic claims, interprets question scopes, and explains the resulting calculations.

## Run

Python 3.11 or later is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
export OPENAI_API_KEY="your-own-api-key"
python main.py "/path/to/questions.json"
```

The default document directory is `documents/` beside the questions file. Input documents must be UTF-8 `.txt` files; subdirectories are included. For the provided local materials:

```bash
python main.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json"
```

Override locations and model settings when needed:

```bash
python main.py "/path/to/questions.json" \
  --documents "/path/to/documents" \
  --db ".backbone/abstraction.sqlite3" \
  --response-dir "responses" \
  --model gpt-5.6-terra \
  --effort medium
```

`OPENAI_MODEL` and `OPENAI_REASONING_EFFORT` provide defaults; command-line arguments take precedence. Effort must be supported by the chosen model. API calls require the runner's own key and API usage budget. Credentials are read from the environment, never from a committed file. `.env.example` documents settings but is not automatically loaded.

The questions format is:

```json
[
  {"id": "Q1", "question": "How much documented therapy did Alex receive?"}
]
```

An object containing a `questions` array is also accepted. IDs must be unique nonempty strings.

## Output

Each invocation creates a new UTC date/time directory and prints the absolute `answers.json` path to standard output. Progress and errors go to standard error.

```text
responses/
  YYYY-MM-DD/
    HHMMSS.microsecondsZ-unique-id/
      answers.json
      abstraction.json
      execution.json
```

`answers.json` contains:

- Run ID, start/end timestamps, status, model settings and configuration hash.
- Absolute source questions path and SHA-256 of the questions file.
- Source document directory and database path.
- Answers keyed by `question_id` within an ordered array, with original question text, interpreted patient/date scope, answer status and limitations.
- Deterministically calculated results and provisional source-document/calculation references.

`abstraction.json` holds the patient abstraction and document source registry, including verified quotation locations. The original document text remains in SQLite. `execution.json` records extraction/cache activity, per-question latency, model requests and reported token usage, total runtime, and database size. Model cost is `null` until a pricing calculation is implemented.

The citation and answer-presentation contract is deliberately provisional. Source IDs, quotes, line/character locations, and calculation references are retained so that the next iteration can specify the desired review format without re-extracting the corpus.

Failures return a nonzero exit code and write a run with `status: failed`; completed answers and prior successful extractions remain available. An empty or partial result must not be treated as a successful clinical answer.

## Reuse and offline runs

The default database is `.backbone/abstraction.sqlite3`, relative to the working directory. Exact document copies reuse an internal document ID and extraction. Changed content creates an immutable revision. Model/prompt/schema settings participate in the cache key. Patient abstractions are reused when their evidence, identity, extraction configuration and reconciliation version are unchanged.

After a successful run, replay the same questions and configuration without model access:

```bash
python main.py "/path/to/questions.json" --offline
```

Offline mode requires cached extractions, question scopes and answers. It fails explicitly on a miss. New questions can reuse the stored clinical abstraction but still need model calls for uncached scope interpretation and narrative generation. `--refresh` recomputes model work and cannot be combined with `--offline`.

Use one database per active document collection. The directory supplied to a run defines its current corpus: removed files stop contributing to that run, while historical document snapshots and caches remain stored. Running a different corpus against the same database replaces its active patient/document index. Moving the same corpus preserves extraction reuse; each run exports current source locations.

## Implementation

- `backbone/storage.py`: SQLite document snapshots, receipts, extraction/cache tables, patient identities, materialized abstractions and patient/date/encounter/document references.
- `backbone/model.py` and `backbone/models.py`: deterministic header candidates, typed Responses API calls, question scopes, evidence-quote validation, and model telemetry.
- `backbone/clinical.py`: encounter reconciliation, scoped corrections, interval arithmetic, uncertainty alternatives, assessment deduplication and weekly plan evaluation.
- `backbone/pipeline.py` and `backbone/cli.py`: ingestion, cache reuse, question execution and timestamped artifacts.

The [design note](docs/design.md) describes the mathematical abstraction. The [standalone API example](examples/ask_model.py) is available for checking model access separately.

## Checks and present limitations

Run offline tests with:

```bash
python -m pytest -q
```

Tests exercise persistence/restart, duplicate and changed documents, scoped identity, evidence rejection, real SDK serialization through a mock HTTP transport, interval calculations, conflicting durations, correction scope, question/answer caches, and timestamped output/failure behavior. Integration tests use invented test records and a deterministic fake model; they are not performance or accuracy measurements on the supplied patient corpus. No live API extraction has been validated yet.

This first implementation uses deterministic header parsing, then one structured semantic extraction for each uncached document. It does not yet have template-specific rules that bypass the model for an entire clinical document. Repeated runs avoid that extraction work. Quote matching verifies that evidence exists; it does not prove each extracted interpretation is correct.

Numerical results in `calculated_results` come from code. The model is instructed to copy them into its prose, and source/calculation IDs are checked, but prose-to-calculation consistency is not independently verified yet. The next citation/output iteration should make numerical bindings explicit.

Supported therapy calculations use patient-present individual, group and family services with Monday–Sunday weeks. Unsupported plan rules, uncertain identity/dates, conflicting accounts and midweek changes are flagged. Overnight time intervals require a future extension. For questions listing separate focus dates, per-date results are supplied; bounding-period results are clearly labeled as including intervening dates. No combined selected-date total is inferred by the model.

Current scaling constraints are serial model extraction on first ingestion, reading every file to detect changed content, rebuilding document-reference indexes, and sending the selected patient's abstraction to each uncached question. A context-size guard fails rather than silently truncating evidence. Profile `execution.json` on a real run before selecting a first bottleneck; likely next steps are a file manifest/change feed, bounded extraction workers, selective evidence retrieval and finer invalidation.

A useful design experiment is to compare this cached structured-extraction baseline against template-specific deterministic extraction with selective LLM fallback, holding reconciliation fixed and measuring source-supported accuracy as well as tokens and latency. That experiment has not yet been run.

## Assistance and reproducibility

Codex assisted with design, implementation, and tests. Runtime defaults are `gpt-5.6-terra` and `medium`; each actual model response records its returned model, usage and request metadata. `requirements.lock.txt` pins the environment tested for this first implementation (Python 3.11, macOS ARM64). Model calls can vary even with identical settings; saved extractions enable exact downstream replay. Before submission, record available coding-assistant model/settings, live corpus results, observed limitations, runtimes and costs.
