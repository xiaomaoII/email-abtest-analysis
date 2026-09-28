"""Read-only stage-three audit; --output creates a new record without overwriting history.

Execute selected saved Notebook cells in memory, recompute subgroup/cost outputs,
and build templates only in a temporary directory. Saved JUnit is evidence from
a separate pytest run; this script never runs pytest or the full pipeline.
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import pandas as pd
from IPython.utils.capture import capture_output
from scipy import stats
from abtest.data import ARMS, prepare, reconcile, sql_metrics
from abtest.notebook import build_notebook

PRACTICE = ["992ae00247d4e0af", "7c5d9288a25c4670", "98388394faf70936", "3998930164f06feb"]
EXECUTE = {"hillstrom-02", "hillstrom-04", "hillstrom-06", "hillstrom-08", "hillstrom-10",
           "hillstrom-14", "hillstrom-16", "hillstrom-18", *[x for x in PRACTICE if x != PRACTICE[1]]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(source, scope, name):
    tree = ast.parse(source)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, name, "exec"), scope)
    if last:
        return eval(compile(ast.Expression(last.value), name, "eval"), scope)


def independent_interaction(df, dimension):
    """Reconstruct the joint test from disjoint cell means, without fitted OLS."""
    cells = df.groupby([dimension, "segment"], observed=True).spend.agg(["mean", "var", "size"])
    keys = list(cells.index)
    levels = sorted(df[dimension].unique())
    assert len(cells) == len(levels) * 3 and (cells["size"] > 1).all()
    contrasts = []
    for level in levels[1:]:
        for arm in ARMS[1:]:
            r = np.zeros(len(cells))
            for key, sign in [((level, arm), 1), ((level, ARMS[0]), -1),
                              ((levels[0], arm), -1), ((levels[0], ARMS[0]), 1)]:
                r[keys.index(key)] = sign
            contrasts.append(r)
    r = np.array(contrasts)
    # Saturated model: each cell mean has HC3 variance s_g²/(n_g-1).
    covariance = r @ np.diag(cells["var"] / (cells["size"] - 1)) @ r.T
    effect = r @ cells["mean"].to_numpy()
    statistic = float(effect @ np.linalg.solve(covariance, effect))
    return {"dimension": dimension, "wald_chi2": statistic, "df": len(contrasts),
            "p_value": float(stats.chi2.sf(statistic, len(contrasts))),
            "total_rows": int(cells["size"].sum()), "minimum_cell_n": int(cells["size"].min())}


def audit(baseline=None):
    os.chdir(ROOT)
    nb_path = ROOT / "notebooks/analysis.ipynb"
    nb_before = digest(nb_path)
    notebook = nbformat.read(nb_path, as_version=4)
    nbformat.validate(notebook)
    assert not [o for c in notebook.cells for o in c.get("outputs", []) if o.output_type == "error"]
    scope, executed = {}, []
    with capture_output():
        for cell in notebook.cells:
            if cell.id in EXECUTE:
                evaluate(cell.source, scope, cell.id)
                executed.append(cell.id)
    assert set(executed) == EXECUTE
    df, raw, config = (scope[k] for k in ["df", "raw", "config"])
    quality, metrics, effects = (scope[k] for k in ["quality", "metrics", "effects"])
    interactions, subgroups, scenarios = (scope[k] for k in ["interactions", "subgroups", "scenarios"])
    assert len(raw) == len(df) == 64000 and quality["zero_spend_rows"] == 63422
    assert quality["exclusions"] == 0
    pd.testing.assert_frame_equal(df, pd.read_csv(ROOT / "data/processed/customers.csv"), check_dtype=False)
    import duckdb
    with duckdb.connect(str(ROOT / "data/processed/hillstrom.duckdb"), read_only=True) as connection:
        database_customers = connection.execute("SELECT * FROM customers ORDER BY source_row_id").df()
    pd.testing.assert_frame_equal(df.sort_values("source_row_id").reset_index(drop=True),
                                  database_customers, check_dtype=False)
    reconcile(df, metrics)
    preview = sql_metrics(df, "SELECT * FROM customers LIMIT 5;")
    pd.testing.assert_frame_equal(preview, df.head(5).reset_index(drop=True))
    changed = raw.copy()
    changed[["visit", "conversion", "spend"]] = 0
    baseline_columns = ["customer_type", "history_band", "recency_band"]
    pd.testing.assert_frame_equal(prepare(changed)[baseline_columns], df[baseline_columns])

    tables = ROOT / "reports/tables"
    matched = []
    for name, frame, keys in [
        ("group_metrics", metrics, ["segment"]),
        ("effects", effects, ["arm", "metric"]),
        ("interaction_tests", interactions, ["dimension"]),
        ("subgroup_effects", subgroups, ["dimension", "subgroup", "arm"]),
        ("profit_scenarios", scenarios, ["arm", "assumed_margin", "assumed_contact_cost_usd"]),
        ("sample_size_plan", scope["planning"], ["mde_usd_per_customer", "sd_multiplier"]),
        ("baseline_balance", scope["balance"], ["arm", "feature"]),
    ]:
        saved = pd.read_csv(tables / f"{name}.csv")
        pd.testing.assert_frame_equal(frame.set_index(keys).sort_index(), saved.set_index(keys).sort_index(),
                                      check_dtype=False, rtol=1e-9, atol=1e-12)
        matched.append(name)

    independent = [independent_interaction(df, dimension) for dimension in baseline_columns]
    for expected in independent:
        actual = interactions.set_index("dimension").loc[expected["dimension"]]
        np.testing.assert_allclose([actual.wald_chi2, actual.p_value, actual["df"]],
                                  [expected["wald_chi2"], expected["p_value"], expected["df"]], rtol=1e-9)
        assert expected["total_rows"] == len(df)
    ordered = interactions.sort_values("p_value")
    holm = np.minimum(1., np.maximum.accumulate(ordered.p_value.to_numpy() * [3, 2, 1]))
    np.testing.assert_allclose(ordered.p_holm_exploratory, holm)
    for row in subgroups.itertuples():
        subset = df[df[row.dimension] == row.subgroup]
        t, c = [subset.loc[subset.segment == arm, "spend"] for arm in [row.arm, ARMS[0]]]
        reference = stats.ttest_ind(t, c, equal_var=False)
        np.testing.assert_allclose([row.effect, row.p_value, row.ci_low, row.ci_high],
                                  [t.mean() - c.mean(), reference.pvalue, *reference.confidence_interval()], rtol=1e-9)
        assert (row.n_treatment, row.n_control) == (len(t), len(c))
        assert (row.buyers_treatment, row.buyers_control) == (int((t > 0).sum()), int((c > 0).sum()))
    segment_sql = sql_metrics(df, (ROOT / "sql/02_segments.sql").read_text(encoding="utf-8"))
    groups = []
    for dimension in baseline_columns:
        g = df.groupby([dimension, "segment"], observed=True).agg(
            n=("spend", "size"), revenue_per_customer=("spend", "mean"), conversion_rate=("conversion", "mean"))
        groups.append(g.reset_index().rename(columns={dimension: "subgroup"}).assign(dimension=dimension))
    keys = ["dimension", "subgroup", "segment"]
    expected = pd.concat(groups).set_index(keys).sort_index()
    for frame in [segment_sql, pd.read_csv(tables / "segment_metrics.csv")]:
        pd.testing.assert_frame_equal(frame.set_index(keys).sort_index()[expected.columns], expected,
                                      check_dtype=False, rtol=1e-9, atol=1e-12)
    assert segment_sql.groupby("dimension").n.sum().eq(len(df)).all()
    matched.append("segment_metrics")

    main = effects[effects.metric == "spend"].set_index("arm")
    for row in scenarios.itertuples():
        income = main.loc[row.arm]
        expected = np.array([income.effect, income.family_ci_low, income.family_ci_high]) * row.assumed_margin - row.assumed_contact_cost_usd
        np.testing.assert_allclose([row.net_per_customer, row.net_ci_low_family, row.net_ci_high_family], expected)
        assert row.break_even_cost_point == income.effect * row.assumed_margin
        assert row.break_even_cost_lower_bound == income.family_ci_low * row.assumed_margin
        signal = "区间下界为正" if expected[1] > 0 else "区间上界为负" if expected[2] < 0 else "区间跨零或触零"
        assert row.scenario_signal == signal
    assert len(scenarios) == 2 * len(config["margins"]) * len(config["contact_costs_usd"]) == 30

    signature = lambda nb: [(c.id, c.cell_type, c.source) for c in nb.cells]
    with tempfile.TemporaryDirectory(prefix="hillstrom-stage3-") as tmp:
        for _ in range(2):
            rebuilt = nbformat.read(build_notebook(Path(tmp), execute=False), as_version=4)
            assert signature(rebuilt) == signature(notebook)
            local = dict(df=df, effects=effects, config=config,
                         profit_scenarios=scope["profit_scenarios"], display=lambda x: None)
            for cell in rebuilt.cells:
                if cell.id in PRACTICE and cell.cell_type == "code":
                    evaluate(cell.source, local, cell.id)
            for key in ["group_means", "women_40"]:
                pd.testing.assert_frame_equal(local[key], scope[key])
            pd.testing.assert_series_equal(local["effect_difference"], scope["effect_difference"])
    assert digest(nb_path) == nb_before

    historical = json.loads((ROOT / "reports/run_manifest.json").read_text(encoding="utf-8"))
    assert digest(ROOT / "data/raw/hillstrom.csv") == historical["source_sha256"]
    assert all(digest(tables / p) == h for p, h in historical["table_sha256"].items())
    if baseline:
        preserved = json.loads(baseline.read_text(encoding="utf-8"))
        preserved.pop("exercises/my_answers.md", None)  # authored responses preserved; Agent status/review updated
    elif (HERE / "audit.json").exists():
        preserved = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))["preserved_sha256"]
    else:
        preserved = {"notebooks/analysis.ipynb": nb_before}
    assert all(digest(ROOT / p) == value for p, value in preserved.items())
    cases = ET.parse(HERE / "tests.xml").getroot().findall(".//testcase")
    assert len(cases) == 41 and all(c.find(tag) is None for c in cases for tag in ["failure", "error", "skipped"])
    inputs = [*ROOT.glob("abtest/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("sql/*.sql"),
              *ROOT.glob("*.md"), ROOT / "config.json", ROOT / "requirements-lock.txt", ROOT / "requirements.txt",
              ROOT / "exercises/my_answers.md", ROOT / "exercises/01_metrics.sql",
              ROOT / "docs/learning_path.md", ROOT / "docs/result_tables.md", Path(__file__), HERE / "review.md"]
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "scope": "Subgroups, conditional cost scenarios, learner draft checkpoint; no next-stage tutoring",
        "python": platform.python_version(), "packages": {p: version(p) for p in historical["packages"]},
        "package_drift_from_historical_lock": {p: {"current": version(p), "locked": v} for p, v in historical["packages"].items() if version(p) != v},
        "source_sha256": digest(ROOT / "data/raw/hillstrom.csv"), "rows": len(df), "zero_spend_rows": quality["zero_spend_rows"],
        "exclusions": quality["exclusions"], "sql_preview_rows": len(preview), "sql_summary_rows": len(metrics),
        "processed_csv_and_readonly_database_match": True,
        "saved_tables_recomputed_and_matched": matched, "segment_totals_per_dimension": segment_sql.groupby("dimension").n.sum().to_dict(),
        "grouping_invariant_to_outcomes": True, "independent_hc3_cell_mean_tests": independent,
        "interactions": interactions.to_dict("records"), "subgroup_rows_checked": len(subgroups),
        "minimum_subgroup_buyers": int(min(subgroups.buyers_treatment.min(), subgroups.buyers_control.min())),
        "customer_type_effects": subgroups[subgroups.dimension == "customer_type"].to_dict("records"),
        "scenario_rows_checked": len(scenarios), "women_margin_40": scenarios[(scenarios.arm == ARMS[2]) & (scenarios.assumed_margin == .4)].to_dict("records"),
        "executed_cell_ids": executed, "notebook_cells": len(notebook.cells), "notebook_written": False,
        "temporary_builds": 2, "generated_ids_order_sources_match": True, "practice_ids": PRACTICE,
        "preserved_sha256": preserved, "historical_table_hashes_match": True,
        "tests": {"passed": len(cases), "report": "reports/checkpoints/2026-09-27/tests.xml", "sha256": digest(HERE / "tests.xml"), "rerun_by_this_script": False},
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs if p.exists()},
        "learning_boundary": "User completed this phase; HC0-3 mastery self-reported, draft exists, resume reference only, no interview rehearsal; await explicit resume",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    baseline = args.baseline.resolve() if args.baseline else None
    if output and output.exists():
        parser.error("Refusing to overwrite an existing audit; choose a new path.")
    payload = json.dumps(audit(baseline), ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"Saved checkpoint: {output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
