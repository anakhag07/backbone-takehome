# Benchmark

Measured October 22, 2026 using `gpt-5.6-terra`, reasoning effort `medium`, on 31 documents totaling 50,106 bytes. The five supplied questions concern one chart patient; one collection-wide question was added. [Recorded metrics](../benchmark-results.json) preserve every attempt, code revision, usage summary, and artifact path.

## Measured results

### Fresh and cached runs

**Fresh documents — initially empty caches; 31 documents**

| Strategy | Document processing | Patient question, mean | Collection question |
| --- | ---: | ---: | ---: |
| Eager | 178.680 s | 12.723 s (5 questions) | 14.714 s |
| Lazy | 0.070 s, text indexing | 76.290 s (5 questions) | 30.395 s |

Lazy's first question took **260.710 s**, including **230.064 s** to parse all 31 documents. It started with **0% parsed**; the remaining questions started with **100% parsed**. Eager parsed everything before answering. Initial processing averaged **5.764 s/document** for eager versus **0.00225 s/document** for lazy's text indexing only.

**Cached documents and answers — exact replay**

| Strategy | Document processing | Patient question, mean | Collection question |
| --- | ---: | ---: | ---: |
| Eager | 0.053 s | 0.0075 s (5 questions) | 0.0084 s |
| Lazy | 0.146 s | 0.0084 s (5 questions) | 0.0085 s |

Replay reuses saved answers and makes no model calls. Lazy uses the latest replay after a citation-display fix; the first complete replay is also retained in the recorded metrics. Both strategies completed all six questions: eager's cold/replay runs took **260.907/0.150 s** overall; lazy's took **411.931/0.201 s**. Completion means processing succeeded: lazy's minute, weekly-goal, and collection answers retain a substantive `partial` status because the evidence does not settle every conclusion. The collection still contains only one chart patient, so cross-patient scaling is unmeasured.

### New answers with documents already parsed

These four questions began with **100% of active and selected documents parsed**, but their answers were uncached.

| Question | Eager | Lazy |
| --- | ---: | ---: |
| DEV-02 | 17.705 s | 29.248 s |
| DEV-03 | 11.077 s | 48.643 s |
| DEV-04 | 11.307 s | 18.893 s |
| DEV-05 | 10.694 s | 23.958 s |
| **Mean** | **12.696 s** | **30.185 s** |

Eager's shared scope planning took 3.81 s and is separate from question latency; lazy includes per-question routing. Lazy DEV-03 includes one rejected citation reference and a successful bounded repair. The earlier two-question warm-document comparison (15.269 s eager; 22.748 s lazy) remains recorded as a separate historical sample.

The saved abstractions contain **400,611 bytes** for eager and **422,572 bytes** for lazy; database files contain **688,128** and **626,688 bytes**, respectively.

## Model usage and cost

| Attempt | Model calls | Mean SDK call time | Input tokens | Output tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager completed cold run | 38 | 6.836 s | 748,371 | 33,477 | $2.12783 |
| Lazy completed cold run | 44 | 9.333 s | 439,392 | 52,954 | $1.72111 |
| Lazy first cold attempt, failed | 33 | 7.534 s | 68,629 | 34,530 | $0.57262 |
| Lazy earlier two-answer continuation, then HTTP 429 | 4, including one failed | 12.286 s | 111,434 reported | 5,979 reported | Unknown total; $0.35033 known subtotal |
| Lazy earlier fresh retry, HTTP 429 | 1, failed | 4.941 s | Unreported | Unreported | Unknown |

The completed lazy run used **31 document readings + 6 routing calls + 6 answers + 1 citation repair**. One answer response failed local reference validation; its successful repair, latency, and cost are included. All 44 calls reported usage. Eager used 62,940 cached input tokens; lazy reported none. A model call measures an SDK invocation, including internal retries; one question can cause several calls. Exact replays used zero calls.

Costs are token-based estimates using [published pricing](https://developers.openai.com/api/docs/models/gpt-5.6-terra), verified October 22, 2026: $2/M input, $0.20/M cached input, and $12/M output, with applicable cache-write and long-context multipliers. Actual billed costs are unavailable. Earlier failed attempts remain separate from the successful-run figures.

## Interruptions and interpretation

Earlier lazy cold attempts failed quotation/reference validation; subsequent fixes resolved citation IDs against source evidence and normalized valid quotation wrappers. An earlier continuation answered two questions before HTTP 429. A new empty-cache attempt at 12:40 UTC also stopped on its first API request, and a minimal diagnostic at 12:49 UTC identified `credit_balance_exhausted`. After credits were added, the **12:55 UTC fresh run completed all six questions**, followed by a successful offline replay. Historical failures and the separate diagnostic remain in the recorded metrics.

Limited source checks of the [completed lazy answers](../results/lazy-answers.json) found:

- Both strategies report **12 sessions/11 days**, treat Jan 21's interrupted contact as one 45-minute encounter, and preserve Jan 26's competing 40/50-minute accounts for one encounter.
- Lazy's collection answer identifies **only Rowan**; eager incorrectly creates Casey, Rowan's partner, as another chart patient.
- Lazy reports **585–595 minutes** overall and **90 minutes on Jan 19** (60 group + 30 individual). Group calculations use attendance intervals minus documented breaks. This assumes the documented scheduled group windows describe delivered activity; eager leaves some group durations unknown. This is an interpretation requiring review, not an indisputable eager arithmetic error.
- Both correctly identify PHQ-9 scores **18, 14, 10** across three administrations and exclude the Jan 26 import as a duplicate of Jan 16. Lazy nevertheless claims recent session material was unavailable despite the supplied Jan 14 record; that limitation needs review.

These are limited spotchecks, not an accuracy score. Source-validated citation lists do not establish support for every narrative claim or guarantee a complete inline citation mapping. The strategies differ in representation and arithmetic as well as scheduling; separate trial times, model/network variation, provider caching, and this small single-patient workload limit generalization. No 500K+ document workload was measured.

## Reproduce or replay

Run both strategies on new databases with automatic offline replays (makes API calls):

```bash
python benchmark.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json"
```

Outputs go to `responses/benchmarks/<timestamp>/`; existing databases are never reset. Replay the completed lazy run using its saved database:

```bash
python main.py responses/benchmarks/lazy-funded-2026-10-22T12-55-14.435507Z-4743fd56/benchmark-questions.json \
  --documents "../Backbone_ Building Clinical Intelligence 09.27/documents" \
  --strategy lazy \
  --db responses/benchmarks/lazy-funded-2026-10-22T12-55-14.435507Z-4743fd56/lazy.sqlite3 \
  --offline
```

This reuses all 31 readings and six answers. Selected result snapshots are checked in; their metadata retains original local paths. Other raw runs and databases are gitignored. Each local run has an `execution-summary.json` with averages, cache percentages, usage, and status.
