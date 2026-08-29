#!/usr/bin/env python
"""CLI: index a folder of documents and print the processing report.

Usage:
    python scripts/ingest.py [--path DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None, help="Documents directory (default: configured documents_dir)")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    pipeline = IngestionPipeline(settings)
    result = pipeline.run(args.path)

    print(json.dumps(result.report.to_dict(), indent=2))
    print(f"\nDone in {result.duration_ms} ms.", file=sys.stderr)

    if result.report.summary()["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
