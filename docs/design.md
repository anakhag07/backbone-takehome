# Initial system design

This design builds on deterministic-first ingestion and reusable extraction. Its main refinement is to separate what a document says from what the system concludes happened. That separation makes duplicate handling, corrections, uncertainty, and numerical answers inspectable. A first CLI implementation now exists; see the README for runnable commands and explicit limitations. The design below also describes intended extensions beyond that first implementation.

## Model documents as evidence about events

Let D be the supplied documents, C their extracted claims, and E the reconstructed care events. The pipeline is:

```text
immutable documents and source spans
                  |
     deterministic extraction + selective LLM extraction
                  |
        typed claims and relationships
                  |
   reconciliation with explicit reasons and alternatives
                  |
   encounters, participation, assessments, effective plans
                  |
     deterministic calculations + evidence-grounded narrative
```

An assertion can be represented as:

`(subject, field, value, source_span, extraction_run, evidence_status)`.

For example, two notes can make competing assertions about the same encounter's start time. Both assertions survive extraction. Reconciliation may accept a scoped correction, recognize a copy, or leave a conflict unresolved. Extraction method and evidentiary authority are separate: a perfectly matched field in a draft note can still be contradicted by a final attendance record.

### A mathematical statement of uncertainty

Let Omega(C) be the nonempty set of plausible event histories retained after applying justified identity, correction, and eligibility rules. Conflicting assertions become alternatives; they are not all imposed as simultaneous hard constraints. For a question q, consider:

`A(q, C) = {q(H) : H in Omega(C)}`.

An answer is established if this set has one value. Otherwise, report the relevant alternatives or conservative bounds and the evidence needed to resolve them. These conclusions are conditional on the available record and stated reconciliation assumptions; they do not bound undocumented care. An empty Omega indicates an inconsistency in reconciliation, not an established answer. A conflict need not invalidate every answer: one encounter can have an established session count but an unresolved duration.

For the prototype, this is a design contract, not a requirement to implement a general constraint solver. Store small explicit alternative sets for material conflicts and propagate them through calculations. Bounds may summarize a result, but a bound such as 40–50 does not imply every intermediate value is independently supported. Preserve correlations when the same ambiguity affects multiple totals.

## Storage boundaries

Use two logical storage structures: a patient abstraction/index keyed by patient ID, and an immutable document store keyed by internal document ID. The patient index holds references to documents, so each body is stored once even when it supports several patients, dates, or encounters.

```text
patients[patient_id]
    identifiers
    by_service_date[date]
        document_ids
        encounters[encounter_id]
            document_ids, claim_ids, resolved_fields, alternatives
    plans, assessments
    unlinked_document_ids

documents[doc_id]
    text, content_hash, source_system, source_document_id
    authored_at, signed_at, received_at
    cached_extractions, source_spans, copy/correction references
```

Define `I(patient, date)` as the set of documents containing evidence about that patient on that service date. This is a retrieval index: membership does not establish that documents describe the same encounter or have equal authority. Retrieve the day's cached claims, distinguish encounters, and reconcile each field using its evidence. Read original source spans or perform further model extraction only when the cached claims do not cover the issue.

One document can belong to several date buckets. A correction received later belongs under the service date it corrects while retaining its receipt date. Preserve unresolved dates and encounter links rather than forcing a bucket. Effective treatment plans and explicit copy/correction links can supply evidence beyond one day's bucket and must remain accessible at patient level. Repeated copies do not become independent votes for a claim.

In Python, these interfaces can look like dictionaries. Persist them in SQLite using a document table, the patient abstraction tables below, and document-reference rows indexed by `(patient_id, service_date, encounter_id, doc_id)`. The index prefix supports patient/day retrieval. Two logical stores need not mean exactly two physical tables or two database services. SQLite supplies persistent indexed lookups; an optional in-memory dictionary cache supplies hash-map lookups without loading all document bodies into memory.

Keep internal `doc_id`, external `(source_system, source_document_id)`, and `content_hash` separate. The document ID identifies an immutable stored revision; the content hash detects identical content for extraction reuse. Hashing a source-assigned document ID does not detect identical content arriving under another ID.

Use an internal patient ID. A name is an attribute, and `(issuing_system, MRN)` is a scoped external identifier. Name and date of birth provide corroborating identity evidence. Missing or conflicting identifiers should create an unresolved linkage rather than force a merge. Keep chart-patient identity distinct from relatives, clinicians, and other participants mentioned in the text.

| Logical entity | Essential information |
| --- | --- |
| Patient and identifiers | Internal patient ID, issuing system, external ID, name/DOB assertions, linkage evidence |
| Document and ingestion receipt | Internal document ID, source document ID, source system/path, content hash, immutable text, authored/signed/received dates, copy relations |
| Source span and assertion | Document revision, line or character offsets, quoted text, subject, typed field/value, extractor run, evidence status |
| Encounter | Internal event ID, scoped encounter/appointment identifiers, service date, service type, participants, attendance state |
| Participation and therapy intervals | Patient presence, therapeutic activity, breaks, interruption intervals, duration assertions and alternatives |
| Assessment and symptom observation | Instrument/form ID, completion date, result, narrative observation, source links |
| Treatment-plan version | Patient/episode, effective interval, eligible services, day/minute thresholds, week definition, source support |
| Resolution and dependency | Accepted/rejected/unresolved assertions, decision reason, correction scope, supporting sources, version |
| Extraction run and coverage | Content/schema/parser/prompt/model versions, fields or sections examined, missing/conflicting results, timings, token use, cost if known |

These are logical boundaries; some can share a table or use validated JSON initially. Keep patient, event, time, and relation fields queryable and indexed. A general-purpose graph database is unnecessary for the first prototype.

Documents and encounters are many-to-many. One register can cover multiple dates and patients; several notes can support one encounter. A document header's date may be its receipt or signature date. Extract each date's role and scope instead of assigning a single generic `date` to all facts.

## Deterministic-first extraction

1. Hash and persist the source; an identical body can reuse extraction while retaining each ingestion receipt.
2. Parse labeled identifiers, explicit dates, table rows, and supported document templates. Validate formats and record exact source spans.
3. Check extraction coverage for each document type: encounters, participants, attendance, intervals, breaks, corrections, assessments, and plan rules as applicable. Successfully parsing a header does not establish that its narrative has been covered.
4. Send uncovered sections or ambiguous fields to an LLM with a typed schema, the deterministic candidates, and enough surrounding text to interpret references. Request source quotes/offsets, explicit missing states, and no invented values. Batch unresolved fields for a document into one request where practical.
5. Validate the output types, source alignment, and semantic consistency. A quote's presence is necessary but does not prove the proposed interpretation. Retain failures, unknown values, and conflicts for review.
6. Reconcile validated claims in a separate step. Calling a model again cannot resolve a genuine disagreement absent new evidence.

Use structured extraction provenance per assertion, such as `method = rule | llm`, `parser_version`, `rule_id`, `model`, `prompt_version`, and `fallback_reason`. Use separate resolution states such as `accepted`, `superseded`, and `unresolved`. Missing, not applicable, not examined, and conflicting must remain distinguishable.

Monitor fallback rate by document template and field, alongside reviewed extraction accuracy and semantic coverage. A high fallback rate suggests where another deterministic rule might help; it does not prove that a rule is safe. Test any replacement against representative labeled cases, including cases the current rules silently miss. Do not treat model self-reported confidence as calibrated correctness.

## Identity and corrections

Exact duplicate documents, semantically repeated evidence, and duplicate descriptions of an encounter are different problems. Hashing solves the first. Explicit copy references, source form identifiers, and shared encounter identifiers help solve the others.

Use explicit encounter/appointment identifiers when available. Otherwise, generate linkage candidates from patient, date/time, service, and participants. Preserve ambiguous linkage rather than merging every same-day service. A phone-call or connection ID can identify a segment of an encounter rather than a separate therapy session.

Corrections apply to a specific subject and field. Do not let a correction to departure time overwrite unrelated clinical content. A later receipt date does not establish greater authority. Preserve original evidence, correction/copy relationships, and the reason for every resolved conflict.

## Calculate care from event identity and time

For each plausible history H, let E_H be its distinct encounters. Let eligible(e) encode the question and applicable treatment-plan definition. Then:

`sessions = sum(1[eligible(e)] for e in E_H)`

`therapy_days = size({service_date(e) : eligible(e)})`

For encounter e, let P_e be patient-presence intervals and T_e be eligible therapeutic activity intervals. The delivered minutes are:

`minutes(e) = measure(P_e intersection T_e)`.

Merge overlapping segments within a consistent account before measuring; exclude documented breaks and technical interruptions. Multiple clinicians in one encounter do not multiply patient minutes. Conflicting accounts of a full encounter remain alternative histories: taking their union would incorrectly choose the longer account. Scalar duration claims can be used when documented, with their own evidence and conflicts.

Weekly totals sum eligible distinct encounters in the documented week convention. If ostensibly separate events overlap, flag the overlap for reconciliation before summing; do not silently inflate total elapsed patient time or assign overlapping minutes to two service types. Divide minutes by 60 only for presentation. Segment intervals at relevant reporting boundaries when necessary.

For a weekly plan predicate g(H), return:

- `met` if g(H) is true in every retained history;
- `not met` if g(H) is false in every retained history;
- `cannot determine` if both outcomes remain possible.

This handles correlated uncertainty more faithfully than independently checking marginal day and minute ranges. The supplied plan's rule is a conjunction: at least three therapy days and at least 150 eligible minutes in a Monday–Sunday week. Extract those values from the source in the implementation; never hardcode this patient's rule or results.

Version plans by effective time rather than ingestion time. For future plan changes, retain the explicit applicability rule; do not invent proration when a midweek change is unspecified. Evaluate two-consecutive-week collection queries using the joint weekly states, identifying both definite and possible inclusion.

## Reuse facts before summaries

Keep a short summary and keywords as a retrieval aid. A configurable budget such as 1,000 characters can bound display/index cost; it must never truncate the underlying evidence, typed claims, or conflict records. Token budgets are more directly useful when preparing model context.

For a known question family Q, the desired abstraction T has the property `q(D) = f_q(T(D))` for q in Q, with evidence and uncertainty preserved. This is a task-specific sufficiency goal, not a guarantee that a bounded prose summary can answer arbitrary future questions.

Answer counts and adherence from structured state. Generate symptom narratives from distinct assessments and cited observations, preserving completion dates, instrument differences, and limitations. A copied measurement is not a new assessment, and absence from a summary does not establish absence from the record.

For a new question, identify its required fields and check recorded extraction coverage. If coverage is incomplete, retrieve relevant source passages, perform targeted extraction, and persist the newly supported claims. Aggregate completeness requires accounting for all relevant records in scope; a top-k retrieval result alone cannot establish a total.

Cache extraction by document content and extraction configuration, including schema/parser/prompt/model versions. Cache reconciled state and answers with their input revisions and rule versions. When new evidence arrives, recompute affected patient/episode events and dependent aggregates, even if the arrival describes an old service date. The smallest safe implementation can rebuild one patient's derived state from persisted claims before introducing finer dependency tracking.

## Corpus cases to use as validation targets

All references below are relative to `../../Backbone_ Building Clinical Intelligence 09.27/documents/` from this file's directory. These are design examples, not patient facts to encode in the implementation.

| Case | Source evidence | Expected behavior |
| --- | --- | --- |
| Scoped correction followed by stale retransmission | `BH-D103_attendance_correction_2026-01-20.txt`, lines 5–13; `BH-D104_resent_roster_received_2026-01-26.txt`, lines 8–20 | Preserve the departure correction despite later receipt of the original roster |
| Two connections in one encounter | `BH-D106_telehealth_2026-01-21.txt`, lines 7 and 15–20 | One encounter, with two therapeutic intervals and an excluded gap |
| Unresolved duration | `BH-D110_individual_primary_record_2026-01-26.txt`, lines 3–7; `BH-D111_individual_second_record_2026-01-26.txt`, lines 4–10 | One encounter with competing 50- and 40-minute accounts; no arbitrary winner |
| Repeated assessment | `imported_measure_summary_received_jan26.txt`, lines 12–19 | Retain the original form/completion date; no new assessment on receipt date |
| Treatment goal | `signed_treatment_plan_jan05.txt`, line 12 | Compile an evidence-linked conjunction of days and minutes with explicit eligibility |

Meaningful checks for implementation should include duplicate invariance, stability under input reordering, reuse after restart, scoped correction effects, source-span validity, interval arithmetic, and preservation of unresolved conflicts. Test related unseen questions and changed document formats. Repeated ingestion may add receipt audit entries while leaving clinical results unchanged.

## Prototype scope and experiment

The first end-to-end CLI uses Python and SQLite and exposes input/output paths, model settings, refresh, and offline mode. Offline tests validate storage, API request wiring, reconciliation and run outputs; live model behavior on the supplied corpus still needs validation. Template-specific semantic parsing, the final citation/presentation contract, broader plan rules and a measured model comparison remain future work.

### Model access and reproducibility

Use a small configurable OpenAI API adapter as the first extraction backend. The [minimal API example](../examples/ask_model.py) demonstrates request wiring only; it does not implement clinical extraction, storage, or reconciliation. It accepts a question plus optional evidence text, reads the API key from the local environment, and makes the model and reasoning effort explicit. The [official quickstart](https://developers.openai.com/api/docs/quickstart) documents SDK installation and environment-based authentication.

An initial model/effort baseline to evaluate is `gpt-5.6-terra` with `medium`; this is a proposed experiment, not an observed quality result. Its [model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-terra) describes the cost/capability positioning and supported efforts. Effort is model-dependent; validate settings if the model changes. The [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) specifies `reasoning={"effort": ...}` for Responses requests.

The actual extraction adapter should use typed [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), validate quoted evidence, and handle refusals and incomplete outputs before writing claims. Schema-conforming output still requires semantic validation. Keep prompt text and schema in version control; keep credentials in the local environment. Include model, effort, prompt/schema versions, source hashes, usage, and raw extraction results in each saved run. Pin tested dependency versions and a model snapshot where available for the final submission; fresh model calls may still vary.

Provide two execution paths in the eventual CLI:

- **Replay:** load a supplied abstraction or extraction cache and reproduce reconciliation/calculations without model access. Cached narrative can be displayed as recorded; new narrative generation may need a model.
- **Fresh ingestion:** process new documents, calling the configured model only when semantic extraction is needed. This requires the runner's own API access, network connection, and usage budget.

GitHub distributes the code and saved artifacts; each runner supplies their own API credentials. Replay is not evidence that a fresh extraction run produces byte-identical output. Fail explicitly on uncached model work in offline mode instead of silently skipping it.

A local open-weight model can be added behind the same extraction contract if offline inference is a requirement or a model-comparison experiment is valuable. It introduces weight downloads, runtime/hardware requirements, and a separate schema/accuracy evaluation. Start with the API so the take-home primarily demonstrates abstraction and reconciliation; keep the extraction interface small enough to replace the backend later.

A useful experiment compares a parse-once structured LLM baseline with deterministic extraction plus selective LLM fallback, holding downstream reconciliation and questions fixed. Measure source-supported accuracy and conflict preservation as well as calls, tokens, cost, and latency. A no-LLM parser is also a useful ablation if time permits. This tests whether rule coverage saves model work without losing semantic facts.

Report initial ingestion time, warm queries, restart queries, targeted updates, individual-patient and collection-wide latency, model usage, and database size. Label estimates separately. With no working pipeline yet, no performance claims are established.

At larger scale, likely bottlenecks depend on measured fallback rates and code paths: model extraction cost if many documents need semantic parsing; candidate explosion if encounter linkage scans all pairs; and repeated full-history scans if query paths lack indexed/materialized state. Keep candidate generation scoped to patient and relevant time windows, index identifiers and effective dates, and reuse persisted claims. Name the first actual bottleneck after profiling the implementation, then describe a concrete migration such as queued extraction workers, shared object storage, or an indexed server database.
