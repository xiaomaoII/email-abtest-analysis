"""Run from any working directory: python /path/to/run.py all."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Hillstrom 营销实验复盘")
    parser.add_argument("command", choices=["all", "fetch", "prepare", "analyze", "notebook", "test"], nargs="?", default="all")
    args = parser.parse_args()
    if args.command == "fetch":
        from abtest.data import download
        print(download(ROOT))
    if args.command in {"prepare", "analyze", "all"}:
        from abtest.pipeline import pipeline
        pipeline(ROOT, prepare_only=args.command == "prepare")
    if args.command in {"notebook", "all"}:
        from abtest.notebook import build_notebook
        build_notebook(ROOT)
    if args.command in {"test", "all"}:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "--junitxml=reports/test_results.xml"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
