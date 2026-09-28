"""Stage-two checkpoint: recompute in memory; never rebuild the user's notebook.

Default: print a fresh audit, leaving saved results unchanged.
--output PATH: create a new audit file exclusively (never overwrite history).
--baseline PATH: optional pre-edit mapping of relative paths to SHA256 hashes.
The saved JUnit file is read as evidence; this script does not run pytest.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import nbformat
import numpy as np
import pandas as pd
from IPython.utils.capture import capture_output
from scipy import stats

from abtest.data import ARMS, reconcile, sql_metrics
from abtest.notebook import build_notebook
from abtest.statistics import bootstrap_difference

STAGE_IDS = {
    "hillstrom-02", "hillstrom-04", "hillstrom-06", "hillstrom-08",
    "hillstrom-10", "c7338eb7ce2c522c", "3ede67f86ce0a3f4",
    "hillstrom-12", "763175ad8d71773c", "hillstrom-18",
}
PRACTICE_IDS = ["c7338eb7ce2c522c", "3ede67f86ce0a3f4", "763175ad8d71773c"]
AUDIT_DIR = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_cell(source, namespace, name):
    """Execute assignments and evaluate the final expression without saving output."""
    tree = ast.parse(source)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, name, "exec"), namespace)
    if last is not None:
        return eval(compile(ast.Expression(last.value), name, "eval"), namespace)


def audit(baseline_path=None):
    os.chdir(ROOT)
    notebook_path = ROOT / "notebooks/analysis.ipynb"
    notebook_hash_before = sha256(notebook_path)
    actual = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(actual)
    assert not any(o.output_type == "error" for c in actual.cells for o in c.get("outputs", []))
    namespace, executed, values = {}, [], {}
    with capture_output():
        for cell in actual.cells:
            if cell.id in STAGE_IDS:
                values[cell.id] = evaluate_cell(cell.source, namespace, cell.id)
                executed.append(cell.id)
    assert len(executed) == len(STAGE_IDS) and set(executed) == STAGE_IDS
    assert values[PRACTICE_IDS[0]] == (4., (-4., 12.))

    raw, df, metrics = (namespace[k] for k in ("raw", "df", "metrics"))
    quality, config, effects, planning = (namespace[k] for k in ("quality", "config", "effects", "planning"))
    srm, balance = namespace["srm"], namespace["balance"]
    reconcile(df, metrics)
    assert len(df) == len(raw) == 64000
    assert quality["zero_spend_rows"] == 63422 and quality["exclusions"] == 0
    assert int(metrics.n.sum()) == len(df)
    preview = sql_metrics(df, "SELECT * FROM customers LIMIT 5;")
    pd.testing.assert_frame_equal(preview, df.head(5).reset_index(drop=True))
    assert len(metrics) == 3

    tables = ROOT / "reports/tables"
    pd.testing.assert_frame_equal(metrics, pd.read_csv(tables / "group_metrics.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    saved = pd.read_csv(tables / "effects.csv")
    pd.testing.assert_frame_equal(effects.set_index(["arm", "metric"]).sort_index(),
                                  saved.set_index(["arm", "metric"]).sort_index(),
                                  check_dtype=False, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(planning, pd.read_csv(tables / "sample_size_plan.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(balance, pd.read_csv(tables / "baseline_balance.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    saved_quality = json.loads((tables / "quality.json").read_text(encoding="utf-8"))
    # The pipeline adds reconciliation status after validate() returns.
    assert quality == {k: saved_quality[k] for k in quality}
    assert saved_quality["sql_python_reconciliation"] == "passed"
    saved_srm = json.loads((tables / "randomization.json").read_text(encoding="utf-8"))
    assert srm["counts"] == saved_srm["counts"]
    np.testing.assert_allclose([srm["p_value"], srm["chi_square"]],
                               [saved_srm["p_value"], saved_srm["chi_square"]], rtol=1e-10, atol=1e-12)
    main = effects[effects.metric == "spend"]
    for row in main.itertuples():
        reference = stats.ttest_ind(df.loc[df.segment == row.arm, "spend"],
                                   df.loc[df.segment == ARMS[0], "spend"], equal_var=False)
        ci = reference.confidence_interval(.95)
        np.testing.assert_allclose([row.p_value, row.ci_low, row.ci_high],
                                   [reference.pvalue, ci.low, ci.high], rtol=1e-10, atol=1e-12)
    ordered = main.sort_values("p_value")
    independent_holm = np.minimum(1., np.maximum.accumulate(ordered.p_value.to_numpy() * [2, 1]))
    np.testing.assert_allclose(ordered.p_holm, independent_holm, rtol=1e-10, atol=1e-12)

    power_checks = []
    critical = stats.norm.isf((config["alpha"] / 2) / 2)
    for row in planning.itertuples():
        assert row.alpha_per_comparison == config["alpha"] / 2
        assert row.n_total_3_arms == row.n_per_arm * 3
        shift = row.mde_usd_per_customer / (row.assumed_sd_usd * np.sqrt(2. / row.n_per_arm))
        attained = float(stats.norm.sf(critical - shift) + stats.norm.cdf(-critical - shift))
        assert config["planning_power"] <= attained < config["planning_power"] + .001
        power_checks.append({"mde": row.mde_usd_per_customer, "sd_multiplier": row.sd_multiplier,
                             "n_per_arm": row.n_per_arm, "n_total_3_arms": row.n_total_3_arms,
                             "normal_approximation_power": attained})

    # Never run execute=True here: stage-three cells must stay unexecuted by this audit.
    with tempfile.TemporaryDirectory(prefix="hillstrom-stage2-") as tmp:
        for _ in range(2):
            rebuilt = nbformat.read(build_notebook(Path(tmp), execute=False), as_version=4)
            nbformat.validate(rebuilt)
            signature = lambda nb: [(c.id, c.cell_type, c.source) for c in nb.cells]
            assert signature(rebuilt) == signature(actual)
            local = {"np": np, "effects": effects, "bootstrap_difference": bootstrap_difference}
            for cell in rebuilt.cells:
                if cell.id in PRACTICE_IDS:
                    result = evaluate_cell(cell.source, local, cell.id)
                    if cell.id == PRACTICE_IDS[0]:
                        assert result == (4., (-4., 12.))
                    else:
                        pd.testing.assert_frame_equal(result, values[cell.id])
            displayed, checked = [], []

            def checked_reconcile(data, summary):
                reconcile(data, summary)
                checked.append(True)

            sql_namespace = {"ROOT": ROOT, "df": df, "display": displayed.append,
                             "sql_metrics": sql_metrics, "reconcile": checked_reconcile}
            with capture_output():
                evaluate_cell(next(c.source for c in rebuilt.cells if c.id == "hillstrom-06"),
                              sql_namespace, "generated-sql")
            assert [len(table) for table in displayed] == [5, 3] and checked == [True]

    historical = json.loads((ROOT / "reports/run_manifest.json").read_text(encoding="utf-8"))
    raw_hash = sha256(ROOT / "data/raw/hillstrom.csv")
    assert raw_hash == historical["source_sha256"]
    assert all(sha256(tables / p) == h for p, h in historical["table_sha256"].items())
    if baseline_path:
        preserved = json.loads(baseline_path.read_text(encoding="utf-8"))
    elif (AUDIT_DIR / "audit.json").exists():
        preserved = json.loads((AUDIT_DIR / "audit.json").read_text(encoding="utf-8"))["preserved_sha256"]
    else:
        preserved = {"notebooks/analysis.ipynb": notebook_hash_before}
    changes = [p for p, h in preserved.items() if sha256(ROOT / p) != h]
    assert not changes, f"Preserved artifacts changed: {changes}"
    assert sha256(notebook_path) == notebook_hash_before

    junit = AUDIT_DIR / "tests.xml"
    cases = ET.parse(junit).getroot().findall(".//testcase")
    failures = [c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
    assert len(cases) == 37 and not failures and not skipped
    inputs = list(historical["inputs_sha256"]) + [
        "exercises/01_metrics.sql", "exercises/my_answers.md", "notebooks/analysis.ipynb",
        "tests/test_analysis.py", "README.md", "PROJECT_STATE.md", "TODO.md", "docs/learning_path.md",
        "reports/checkpoints/2026-09-24/verify.py",
    ]
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stage two through sample-size planning; no stage-three analysis or report refresh",
        "python": platform.python_version(),
        "packages": {p: version(p) for p in historical["packages"]},
        "package_drift_from_historical_lock": {p: {"current": version(p), "locked": v}
                                                for p, v in historical["packages"].items() if version(p) != v},
        "source_sha256": raw_hash, "rows": len(df), "zero_spend_rows": quality["zero_spend_rows"],
        "exclusions": quality["exclusions"], "preview_rows": len(preview), "summary_rows": len(metrics),
        "sql_python_reconciliation": True, "effects_match_saved": True, "planning_matches_saved": True,
        "srm": srm, "max_abs_smd": float(balance.abs_smd.max()),
        "main_effects": main[["arm", "effect", "p_value", "p_holm", "ci_low", "ci_high",
                               "bootstrap_low", "bootstrap_high", "family_ci_low", "family_ci_high"]].to_dict("records"),
        "planning": planning.to_dict("records"), "planning_power_checks": power_checks,
        "toy_bootstrap": {"effect": 4., "ci": [-4., 12.]},
        "executed_current_stage_cell_ids": executed, "notebook_written": False,
        "notebook_cells": len(actual.cells), "temporary_builds": 2,
        "generated_ids_order_sources_match_notebook": True, "generated_preview_and_summary_rows": [5, 3],
        "preserved_sha256": preserved, "preserved_files_unchanged": True,
        "historical_tables_hashes_match": True,
        "historical_manifest": "reports/run_manifest.json (2026-09-16); not rewritten",
        "historical_checkpoint": "reports/checkpoint.json (2026-09-20); not rewritten",
        "tests": {"command": r".\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/checkpoints/2026-09-24/tests.xml",
                  "report": str(junit.relative_to(ROOT)).replace("\\", "/"), "passed": len(cases),
                  "sha256": sha256(junit), "rerun_by_this_script": False},
        "inputs_sha256": {p: sha256(ROOT / p) for p in inputs},
        "learning_boundary": "Practice exists; A/A, power meaning and final-cohort 14-day window await learner confirmation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    baseline = args.baseline.resolve() if args.baseline else None
    if output and output.exists():
        parser.error("Output already exists; use a new path to preserve historical evidence.")
    record = audit(baseline)
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        print(f"Saved checkpoint: {output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
