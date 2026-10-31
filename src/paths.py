"""Central repository paths, independent of working directory and module depth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARCHIVE = ROOT / "archive"
DOCS = ROOT / "docs"
