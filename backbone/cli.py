"""Command-line entry point."""

import argparse
import os
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Answer questions using a persistent clinical abstraction. "
        "Documents default to the documents/ directory beside questions.json."))
    parser.add_argument("questions", type=Path, help="Path to a questions.json file")
    parser.add_argument("--documents", type=Path, help="Directory containing UTF-8 .txt records")
    parser.add_argument("--db", type=Path, default=Path(".backbone/abstraction.sqlite3"))
    parser.add_argument("--response-dir", type=Path, default=Path("responses"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"))
    parser.add_argument("--effort", default=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
                        help="Reasoning effort supported by the selected model")
    parser.add_argument("--offline", action="store_true", help="Use saved extraction, scope, and answer caches only")
    parser.add_argument("--refresh", action="store_true", help="Recompute model work instead of reading caches")
    parser.add_argument("--max-context-chars", type=int, default=400_000,
                        help="Maximum serialized context characters per question (default: %(default)s)")
    args = parser.parse_args(argv)
    from .pipeline import Settings, run_pipeline

    settings = Settings(questions=args.questions.expanduser().resolve(),
        documents=(args.documents or args.questions.parent / "documents").expanduser().resolve(),
        database=args.db.expanduser().resolve(), responses=args.response_dir.expanduser().resolve(),
        model=args.model, effort=args.effort, offline=args.offline, refresh=args.refresh,
        max_context_chars=args.max_context_chars)
    try:
        result = run_pipeline(settings, progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (OSError, ValueError) as error:
        print(f"Unable to create response run: {error}", file=sys.stderr)
        return 1
    print(result.answers_path)
    if result.error:
        print(result.error, file=sys.stderr)
    return 0 if result.status == "completed" else 1
