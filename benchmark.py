"""Compare both strategies on fresh databases, then replay without network access."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

from src.deferred import run_deferred_pipeline
from src.pipeline import Settings, _write_json, load_questions, run_pipeline
from src.reporting import make_run_id

COLLECTION_QUESTION = {
    "id": "BENCH-COLLECTION",
    "question": "For every patient represented in the supplied collection, report the number of eligible patient-present therapy sessions, distinct therapy days, and therapy minutes for January 5–30, 2026, separately by patient. Preserve any unresolved alternatives and cite supporting sources.",
}


def run_benchmark(questions_path, documents, output, model, effort, max_context_chars=400_000):
    output.mkdir(parents=True, exist_ok=False)
    questions, questions_hash = load_questions(questions_path)
    if any(q["id"] == COLLECTION_QUESTION["id"] for q in questions):
        raise ValueError("The benchmark's supplemental question ID already exists.")
    augmented = output / "benchmark-questions.json"
    _write_json(augmented, questions + [COLLECTION_QUESTION])
    manifest = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                 "bytes": p.stat().st_size} for p in sorted(documents.rglob("*")) if p.is_file() and p.suffix.lower() == ".txt"]
    comparison = {"measurement_type": "single_trial_live_api", "model": model, "effort": effort,
        "source_questions": {"path": str(questions_path), "sha256": questions_hash},
        "documents": manifest, "document_count": len(manifest), "source_bytes": sum(d["bytes"] for d in manifest),
        "supplemental_question": COLLECTION_QUESTION,
        "method": "Same augmented question order/model/effort. Separate fresh SQLite databases for each strategy. Each cold run is followed by an offline replay using its own saved cache.",
        "limitations": ["One timing trial per strategy; order, network variation, API prompt caching and model variability affect results.",
            "Cold means an empty application database; provider-side prompt caching is not controlled.",
            "Supplied documents contain one chart patient; the collection question does not measure multi-patient scale.",
            "Eager uses a clinical schema and Python arithmetic; lazy uses generic notes and model reasoning. This is a design comparison, not an isolated scheduling experiment or proof of equal accuracy.",
            "No manually labeled answer benchmark is supplied; output disagreements need review."], "runs": []}
    for strategy, runner in (("eager", run_pipeline), ("lazy", run_deferred_pipeline)):
        settings = Settings(questions=augmented, documents=documents, database=output / f"{strategy}.sqlite3",
            responses=output / strategy, model=model, effort=effort, max_context_chars=max_context_chars)
        for temperature in ("cold", "replay"):
            print(f"[{strategy} {temperature}]", file=sys.stderr, flush=True)
            current = replace(settings, offline=temperature == "replay")
            result = runner(current, progress=lambda message: print(message, file=sys.stderr, flush=True))
            summary_path = result.answers_path.with_name("execution-summary.json")
            comparison["runs"].append({"strategy": strategy, "cache_state": temperature,
                "status": result.status, "error": result.error,
                "answers_path": str(result.answers_path), "summary_path": str(summary_path),
                "summary": json.loads(summary_path.read_text())})
            _write_json(output / "comparison.json", comparison)
            if result.status != "completed":
                print(f"{strategy} stopped: {result.error}", file=sys.stderr, flush=True)
                break
    return output / "comparison.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--output", type=Path, help="New directory; existing directories are never reset")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"))
    parser.add_argument("--effort", default=os.getenv("OPENAI_REASONING_EFFORT", "medium"))
    parser.add_argument("--max-context-chars", type=int, default=400_000)
    args = parser.parse_args()
    path = run_benchmark(args.questions.expanduser().resolve(),
        (args.documents or args.questions.parent / "documents").expanduser().resolve(),
        (args.output or Path("responses/benchmarks") / make_run_id()).expanduser().resolve(),
        args.model, args.effort, args.max_context_chars)
    print(path)
    return int(any(run["status"] != "completed" for run in json.loads(path.read_text())["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
