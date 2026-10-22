# Benchmark

Measured October 22, 2026 using `gpt-5.6-terra`, reasoning effort `medium`, on 31 documents totaling 50,106 bytes. The corpus contains one chart patient. The five supplied questions are individual-patient questions; one explicitly labeled collection-wide question was added. [Machine-readable results](../benchmark-results.json) retain every attempt, code revision, usage record summary, and artifact path.

## Measured results

### Fresh and cached runs

**Fresh documents — initially empty caches; 31 documents**

| Strategy | Document processing | Patient question, mean | Collection question |
| --- | ---: | ---: | ---: |
| Eager | 178.680 s | 12.723 s (5 questions) | 14.714 s |
| Lazy | 0.061 s, text indexing | Unavailable | Unavailable |

**Cached documents and answers — exact replay**

| Strategy | Document processing | Patient question, mean | Collection question |
| --- | ---: | ---: | ---: |
| Eager | 0.053 s | 0.0075 s (5 questions) | 0.0084 s |
| Lazy | 0.187 s | 0.0084 s (2 questions) | Unavailable |

Replay reuses saved answers. Lazy runs were incomplete; the collection contains one chart patient, so cross-patient performance is unmeasured. Cached figures come from eager replay and lazy partial offline replay in [the recorded metrics](../benchmark-results.json).

### New-answer comparison

These are observed phase timings from the runs described below, **not two successful cold end-to-end runs**.

| Measurement | Eager | Deferred (`lazy`) |
| --- | ---: | ---: |
| Fresh initial processing | 178.680 s | 0.061 s, text index only |
| Initial processing per document | 5.764 s | 0.00196 s |
| DEV-01, uncached answer with documents already parsed | 12.833 s | 18.092 s |
| DEV-02, uncached answer with documents already parsed | 17.705 s | 27.404 s |
| Mean of those same two questions | **15.269 s** | **22.748 s** |
| Mean across all five supplied questions | 12.723 s | Unavailable: only two completed |
| Supplemental collection-wide question | 14.714 s | Unavailable: HTTP 429 stopped the run earlier |
| Saved abstraction after all document readings | 400,611 bytes | 419,330 bytes |

All completed questions above started with **100% of active and selected documents parsed**. Eager shared scope planning took 3.81 s and is separate from question latency. Lazy includes routing in question latency; DEV-01's route was already cached from an earlier attempt. Lazy routing for DEV-02 was uncached. This comparison includes those implementation differences.

Eager completed its cold run in **260.907 s**. Its exact six-question offline replay took **0.150 s** with zero model calls. Lazy replayed its two saved answers with zero calls, then correctly stopped on the missing DEV-03 route; it has no complete six-question replay measurement.

## Model usage and cost

| Attempt | Model calls | Mean SDK call time | Input tokens | Output tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager completed cold run | 38 | 6.836 s | 748,371 | 33,477 | $2.12783 |
| Lazy first cold attempt, failed | 33 | 7.534 s | 68,629 | 34,530 | $0.57262 |
| Lazy two-answer continuation, then HTTP 429 | 4, including one failed | 12.286 s | 111,434 reported | 5,979 reported | Unknown total; $0.35033 known subtotal |

Eager used 62,940 cached input tokens. The lazy continuation's three successful calls averaged **14.956 s**; its fourth call returned no usage, so the full cost is `null`. A model call is one measured SDK invocation, including SDK retry time; it is not necessarily one HTTP attempt. A question may cause routing, several document readings, answering, an original-source reread, or citation repair. Exact cached answers need no model calls.

Costs are token-based estimates using [published pricing](https://developers.openai.com/api/docs/models/gpt-5.6-terra), verified October 22, 2026: $2/M input, $0.20/M cached input, and $12/M output, with the cache-write and long-context multipliers when applicable. Actual billed costs are unavailable. The eager five-question answer stage cost an estimated $1.44989, or $0.29/question, excluding intake and shared planning. Each attempt's separate cost is retained; failed attempts are not silently included in a successful-run figure.

## Interruptions and interpretation

The first lazy cold run indexed text in 0.061 s, then spent **219.971 s** reading all 31 documents for DEV-01. Its **248.850 s attempted-question duration ended in citation validation failure** and is not a completed-answer latency. A quotation-copy correction attempt also failed. The answer adapter now selects labeled evidence IDs; Python resolves them into source quotations. A second fresh run stopped during document reading because of an added quotation wrapper; deterministic normalization now removes only a matching wrapper whose inner text still occurs in the source.

The repaired implementation reused the first run's 31 valid cached readings and completed DEV-01 and DEV-02. HTTP 429 stopped DEV-03. A later resumed attempt also returned HTTP 429, so further API requests were stopped. Neither a successful cold lazy first-answer latency nor the remaining lazy question latencies have been inferred from these fragments. No runtime estimate is substituted for a measurement.

Limited source checks found:

- Both observed strategies report 12 sessions and 11 days, retain Jan 26 as one encounter with 40/50-minute alternatives, and count Jan 21's interrupted contact as one 45-minute encounter.
- Eager incorrectly creates Casey, Rowan's partner, as another chart patient in the collection answer. The lazy collection answer remains unavailable.
- Lazy calculates Jan 19 as 60 group minutes plus 30 individual minutes. This depends on treating the documented scheduled group window as the activity window. Eager leaves group minutes unknown; this is an interpretation requiring review, not an indisputable arithmetic error.
- Eager's PHQ-9 output matches three assessment occasions with scores 18, 14 and 10. Lazy DEV-05 remains unavailable.

These are limited spotchecks, not an accuracy score. The strategies differ in representation and arithmetic as well as scheduling. Provider prompt caching, network/model variation, code repairs, and the small single-patient workload limit generalization. The README identifies the full-corpus scan, serial calls, and catalog/context construction that constrain 500K+ documents; that scale has not been measured.

## Reproduce or resume

Run both strategies on new databases with automatic offline replays (makes API calls):

```bash
python benchmark.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json"
```

Outputs go to `responses/benchmarks/<timestamp>/`; existing databases are never reset. To resume this recorded partial experiment after API access permits more requests, reuse the exact augmented question file and saved database:

```bash
python main.py responses/benchmarks/comparison-2026-10-22-live/benchmark-questions.json \
  --documents "../Backbone_ Building Clinical Intelligence 09.27/documents" \
  --strategy lazy \
  --db responses/benchmarks/comparison-2026-10-22-live/lazy.sqlite3
```

This reuses all 31 readings and the first two answers. Raw local runs and databases are gitignored. The checked-in JSON preserves compact measured results; each local run has its own `execution-summary.json` with averages, cache percentages, usage, and status.
