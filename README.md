# Backbone take-home

A prototype for deriving a persistent, auditable representation of care from clinical records and answering questions with reproducible calculations.

Current status: repository initialized for design and implementation. The [initial design](docs/design.md) is a proposal; ingestion, extraction, reconciliation, and question answering are not implemented yet.

A standalone [API request example](examples/ask_model.py) demonstrates model access; it is not connected to the proposed clinical pipeline.

The central design is to preserve source-linked assertions, reconcile them into distinct care events, and compute answers over those events. Deterministic extraction runs first; a structured LLM extraction step handles uncovered semantic content. Summaries help find relevant evidence, while structured events support numerical answers and explicit uncertainty.

## Inputs

The supplied materials remain in the sibling directory:

```text
../Backbone_ Building Clinical Intelligence 09.27/
    Problem Statement.docx
    questions.json
    documents/
```

The development questions concern one patient's episode. The broader problem statement also describes collection-wide queries and treatment-plan changes; keep those dimensions in the representation without expanding the prototype into a universal clinical system.

## Proposed implementation

Start with Python, SQLite, and a CLI. Keep the first end-to-end path small:

1. Persist immutable documents and line-addressable source text with content hashes.
2. Extract typed assertions and record field-level evidence and extraction provenance.
3. Reconcile encounters, attendance, therapeutic intervals, assessments, and plan requirements.
4. Calculate counts, days, minutes, and plan adherence in code; report unresolved alternatives.
5. Produce inspectable answers and measure initial processing, reuse after restart, model use, and saved-state size.

See the design note for schema boundaries, formulas, cache behavior, and proposed validation cases. No benchmark results or cost claims have been measured yet.

## Try an API request

Run from the repository root with Python 3 in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install openai
export OPENAI_API_KEY="your-own-api-key"
python examples/ask_model.py "Explain the difference between a document and an encounter." --model gpt-5.6-terra --effort medium
```

The example also accepts `--document PATH` to supply a UTF-8 text file as evidence. `OPENAI_MODEL` and `OPENAI_REASONING_EFFORT` set defaults; command-line flags override them. Supported effort values depend on the selected model. API requests use the runner's own account and incur usage charges. Keep the key out of Git. The dependency is unpinned during this design stage; record a tested version before submission.

Only offline syntax, help, and missing-key behavior have been checked for this example. A live API call has not been run. Replay from a saved clinical abstraction remains planned work.

## Assistance

Codex assisted with repository setup and this initial design discussion. Before submission, record the actual coding-assistant model/settings available to the author, extraction-model names/settings, runtime measurements, model usage, and costs. Do not present design estimates as measured results.
