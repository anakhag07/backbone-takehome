# Benchmark

Measured October 22, 2026: 31 documents, one chart patient, five supplied questions plus one collection-wide question; `gpt-5.6-terra`, effort `medium`.

| Measurement | Eager, fresh database | Eager, offline replay | Lazy, fresh database |
| --- | ---: | ---: | ---: |
| Initial processing | 178.68 s | 0.053 s | 0.061 s (text index only) |
| Initial processing per document | 5.76 s | 0.0017 s | 0.0020 s |
| Shared question-scope planning | 3.81 s | 0.0002 s | Included in question |
| Mean individual-patient question (5) | 12.72 s | 0.0075 s | No completed answers |
| Collection-wide question (1) | 14.71 s | 0.0084 s | Not reached |
| Total run | 260.91 s | 0.15 s | 248.91 s (failed) |
| Model calls | 38 | 0 | 33 |
| Estimated model cost, entire run | $2.13 | $0 | $0.57 (failed) |

Sources: [eager summary](../responses/benchmarks/comparison-2026-10-22-live/eager/2026-10-22/2026-10-22T11-10-00.414182Z-44acc4e3/execution-summary.json), [replay summary](../responses/benchmarks/comparison-2026-10-22-live/eager/2026-10-22/2026-10-22T11-14-21.349286Z-4c80de08/execution-summary.json), [lazy summary](../responses/benchmarks/comparison-2026-10-22-live/lazy/2026-10-22/2026-10-22T11-14-21.532293Z-1da56b8d/execution-summary.json), and [comparison manifest](../responses/benchmarks/comparison-2026-10-22-live/comparison.json).

The eager cold run used 748,371 input tokens (62,940 cached) and 33,477 output tokens. Saved abstraction: **400,611 bytes**; database: **688,128 bytes**.

**Question cost:** $1.4498865 / 5 = **$0.29 per supplied question**; including the supplemental collection question, $1.78558 / 6 = **$0.30**. These exclude intake and shared planning. Estimates use reported token usage and [published pricing](https://developers.openai.com/api/docs/models/gpt-5.6-terra): $2/M input, $0.20/M cached input, $12/M output, with the 1.25× cache-write input charge where reported. No answer exceeded the 272K-input-token pricing threshold. Actual billed cost is unavailable.

Eager intake includes extraction and initial abstraction; shared scope planning is separate from question latency. Lazy's first question spent **219.97 s** parsing all 31 documents, then failed because a citation quotation did not occur in the supplied evidence. Its **248.85 s** attempted-question time is not a successful-answer latency; no replay followed.

Replay reuses exact answers and does not measure new questions. These are single trials with uncontrolled provider caching and model/network variability. The collection question covers only one patient. The strategies also differ in schema and arithmetic, so this does not establish equal accuracy or a successful speed comparison.

To rerun both strategies with fresh databases and offline replays (makes API calls):

```bash
python benchmark.py "../Backbone_ Building Clinical Intelligence 09.27/questions.json"
```

Outputs go to `responses/benchmarks/<timestamp>/`. Generated responses/databases are gitignored; select the reviewed submission artifacts later.
