"""Tests target estimands, statistical checks and concrete failure modes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtest.data import ARMS, download, prepare, reconcile, sql_metrics, validate
from abtest.statistics import (bootstrap_difference, profit_scenarios, randomization_checks,
                               recommendation, relative_lift, sample_size_plan,
                               validate_config, welch)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


@pytest.fixture
def customers():
    rows = []
    for arm in ARMS:
        for spend in [0., 0., 10., 20.]:
            rows.append({"recency": 3, "history_segment": "1) $0 - $100", "history": 100.,
                         "mens": 1, "womens": 0, "zip_code": "Surburban", "newbie": 0,
                         "channel": "Web", "segment": arm,
                         "visit": int(spend > 0), "conversion": int(spend > 0), "spend": spend})
    return pd.DataFrame(rows)


def test_denominator_includes_zero_spend_and_identical_rows(customers):
    quality = validate(customers)
    prepared = prepare(customers)
    metrics = sql_metrics(prepared, (ROOT / "sql/01_metrics.sql").read_text(encoding="utf-8"))
    assert quality["identical_rows_after_first"] == 3
    assert len(prepared) == 12
    assert (metrics.n == 4).all()
    assert (metrics.revenue_per_customer == 7.5).all()
    assert (metrics.conversion_rate == .5).all()
    reconcile(prepared, metrics)


def test_exercise_sql_matches_independent_metrics(customers):
    prepared = prepare(customers)
    actual = sql_metrics(prepared, (ROOT / "exercises/01_metrics.sql").read_text(encoding="utf-8"))
    reconcile(prepared, actual)
    assert len(actual) == 3
    assert actual.n.sum() == len(customers)


def test_rebuilt_notebook_displays_preview_and_exercise_summary(tmp_path, monkeypatch, customers):
    import nbformat
    from abtest.notebook import build_notebook

    for variable in ["JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "IPYTHONDIR"]:
        monkeypatch.delenv(variable, raising=False)
    exercise = tmp_path / "exercises/01_metrics.sql"
    exercise.parent.mkdir()
    # A valid extra column proves this reads the exercise, not the reference query.
    query = (ROOT / "exercises/01_metrics.sql").read_text(encoding="utf-8")
    exercise.write_text(query.replace("COUNT(*) AS n", "42 AS exercise_marker, COUNT(*) AS n"), encoding="utf-8")
    prepared = prepare(customers)
    for _ in range(2):
        notebook = nbformat.read(build_notebook(tmp_path, execute=False), as_version=4)
        cell = next(c for c in notebook.cells if c.id == "hillstrom-06")
        displays, checks = [], []

        def checked_reconcile(df, metrics):
            reconcile(df, metrics)
            checks.append(True)

        namespace = {"ROOT": tmp_path, "df": prepared, "sql_metrics": sql_metrics,
                     "reconcile": checked_reconcile, "display": displays.append}
        exec(compile(cell.source, "rebuilt-notebook-sql", "exec"), namespace)
        assert len(displays) == 2
        pd.testing.assert_frame_equal(displays[0], prepared.head(5).reset_index(drop=True))
        assert len(displays[1]) == 3
        assert (displays[1].exercise_marker == 42).all()
        assert checks == [True]
        assert namespace["metrics"] is displays[1]


def test_sql_reconciliation_detects_wrong_denominator(customers):
    metrics = sql_metrics(customers, (ROOT / "sql/01_metrics.sql").read_text(encoding="utf-8"))
    metrics["revenue_per_customer"] = 15.
    with pytest.raises(AssertionError):
        reconcile(customers, metrics)


def test_rebuilt_notebook_preserves_and_runs_stage_two_exercises(tmp_path, monkeypatch):
    import ast
    import nbformat
    from abtest.notebook import build_notebook

    for variable in ["JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "IPYTHONDIR"]:
        monkeypatch.delenv(variable, raising=False)
    practice_ids = ["c7338eb7ce2c522c", "3ede67f86ce0a3f4", "763175ad8d71773c"]
    expected_ids = [f"hillstrom-{i:02d}" for i in range(22)]
    expected_ids[11:11] = practice_ids[:2]
    expected_ids.insert(expected_ids.index("hillstrom-13"), practice_ids[2])
    expected_ids[expected_ids.index("hillstrom-15"):expected_ids.index("hillstrom-15")] = [
        "992ae00247d4e0af", "7c5d9288a25c4670", "98388394faf70936"]
    expected_ids.insert(expected_ids.index("hillstrom-17"), "3998930164f06feb")
    # Shuffled arms and a secondary metric catch missing filtering/sorting.
    effects = pd.DataFrame({
        "arm": [ARMS[2], ARMS[1], ARMS[1]],
        "metric": ["spend", "visit", "spend"],
        "effect": [.4, .08, .8], "ci_low": [.1, .06, .5],
        "ci_high": [.7, .1, 1.1], "bootstrap_low": [.12, np.nan, .52],
        "bootstrap_high": [.69, np.nan, 1.09],
        "p_value": [.001, .000001, .0001], "p_holm": [.001, np.nan, .0002],
    })
    for _ in range(2):
        notebook = nbformat.read(build_notebook(tmp_path, execute=False), as_version=4)
        nbformat.validate(notebook)
        assert [c.id for c in notebook.cells] == expected_ids
        namespace = {"np": np, "effects": effects, "bootstrap_difference": bootstrap_difference}
        results = []
        for cell in notebook.cells:
            if cell.id not in practice_ids:
                continue
            # Evaluate the last expression as a Notebook would, not just its assignments.
            tree = ast.parse(cell.source)
            last = tree.body.pop()
            assert isinstance(last, ast.Expr)
            exec(compile(tree, cell.id, "exec"), namespace)
            results.append(eval(compile(ast.Expression(last.value), cell.id, "eval"), namespace))
        assert results[0] == (4., (-4., 12.))
        pd.testing.assert_frame_equal(results[1], effects.loc[[0, 2],
            ["arm", "effect", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high"]])
        pd.testing.assert_frame_equal(results[2], effects.loc[[2, 0], ["arm", "p_value", "p_holm"]])


def test_missing_is_not_zero(customers):
    customers.loc[0, "spend"] = np.nan
    with pytest.raises(ValueError, match="缺失"):
        validate(customers)


def test_empty_group_is_rejected(customers):
    with pytest.raises(ValueError, match="三个实验组"):
        validate(customers[customers.segment != ARMS[1]])


def test_empty_data_is_rejected(customers):
    with pytest.raises(ValueError, match="为空"):
        validate(customers.iloc[:0])


@pytest.mark.parametrize("column,value", [("conversion", 2), ("spend", -1), ("history", np.inf), ("recency", 2.5), ("channel", "Unknown")])
def test_invalid_inputs_fail(customers, column, value):
    customers[column] = customers[column].astype(object) if column == "channel" else customers[column].astype(float)
    customers.loc[0, column] = value
    with pytest.raises(ValueError):
        validate(customers)


def test_inconsistent_outcome_is_rejected(customers):
    customers.loc[0, "spend"] = 5
    with pytest.raises(ValueError, match="矛盾"):
        validate(customers)


def test_source_values_untouched_and_bands_correct(customers):
    customers.loc[:3, "history"] = [100, 100.01, 350, 350.01]
    customers.loc[:3, "recency"] = [3, 4, 6, 7]
    prepared = prepare(customers)
    assert list(prepared.history_band.iloc[:4]) == ["≤100美元", "100–350美元", "100–350美元", ">350美元"]
    assert list(prepared.recency_band.iloc[:4]) == ["1–3个月", "4–6个月", "4–6个月", "7–12个月"]
    assert set(customers.zip_code) == {"Surburban"}
    assert set(prepared.zip_code) == {"Suburban"}


def test_welch_matches_scipy_unequal_variances():
    treatment, control = [0, 0, 10, 100, 3], [0, 0, 0, 2, 5, 6]
    actual = welch(treatment, control)
    reference = stats.ttest_ind(treatment, control, equal_var=False)
    interval = reference.confidence_interval(.95)
    assert actual["p_value"] == pytest.approx(reference.pvalue)
    assert actual["df"] == pytest.approx(reference.df)
    assert actual["ci_low"] == pytest.approx(interval.low)
    assert actual["ci_high"] == pytest.approx(interval.high)


def test_interval_symmetry_under_group_swap():
    forward = welch([0, 0, 30, 80], [0, 0, 4, 10, 0])
    reverse = welch([0, 0, 4, 10, 0], [0, 0, 30, 80])
    assert forward["effect"] == pytest.approx(-reverse["effect"])
    assert forward["ci_low"] == pytest.approx(-reverse["ci_high"])
    assert forward["p_value"] == pytest.approx(reverse["p_value"])


def test_zero_baseline_is_unavailable():
    assert np.isnan(relative_lift(1, 0))
    assert relative_lift(3, 2) == .5


@pytest.mark.parametrize("treatment", [[], [1], [1, np.nan], [1, np.inf]])
def test_welch_rejects_unusable_groups(treatment):
    with pytest.raises(ValueError):
        welch(treatment, [0, 1, 2])


def test_constant_groups_do_not_claim_significance():
    result = welch([1, 1, 1], [0, 0, 0])
    assert result["status"] == "zero_variance"
    assert np.isnan(result["p_value"])


def test_bootstrap_reproducible_including_zeros():
    sample = [0, 0, 0, 10, 20]
    a = bootstrap_difference(sample, sample, 400, 42)
    b = bootstrap_difference(sample, sample, 400, 42)
    assert a == b
    assert a[0] < 0 < a[1]


def test_srm_detects_imbalanced_assignment(customers):
    balanced, _ = randomization_checks(customers, CONFIG)
    assert not balanced["flagged"]
    unbalanced = pd.concat([customers] + [customers[customers.segment == ARMS[1]]] * 100, ignore_index=True)
    flagged, _ = randomization_checks(unbalanced, CONFIG)
    assert flagged["flagged"]


def test_profit_uses_revenue_times_margin_minus_cost():
    effect = pd.DataFrame([dict(arm=ARMS[1], metric="spend", effect=.5, family_ci_low=.1, family_ci_high=.9)])
    config = {**CONFIG, "margins": [.4], "contact_costs_usd": [.05]}
    result = profit_scenarios(effect, config).iloc[0]
    assert result.net_per_customer == pytest.approx(.15)
    assert result.net_ci_low_family == pytest.approx(-.01)
    assert result.net_ci_high_family == pytest.approx(.31)
    assert result.scenario_signal == "区间跨零或触零"


def test_mde_halving_quadruples_sample_size():
    config = {**CONFIG, "planning_mde_usd": [.5, 1.]}
    result = sample_size_plan(pd.DataFrame({"spend_variance": [225., 400., 256.]}), config)
    subset = result[result.sd_multiplier == 1].set_index("mde_usd_per_customer")
    assert abs(subset.loc[.5, "n_per_arm"] - 4 * subset.loc[1., "n_per_arm"]) <= 3
    assert (result.n_total_3_arms == 3 * result.n_per_arm).all()


def test_sample_size_attains_power_at_family_adjusted_two_sided_threshold():
    config = {**CONFIG, "planning_mde_usd": [.5, 1.]}
    result = sample_size_plan(pd.DataFrame({"spend_variance": [225., 400., 256.]}), config)
    # Independently evaluate both tails of the normal rejection region.
    # The comparison threshold is derived from the design, not from the output.
    alpha_pair = config["alpha"] / 2
    critical = stats.norm.isf(alpha_pair / 2)
    for row in result.itertuples():
        sd = 20. * row.sd_multiplier
        assert row.assumed_sd_usd == pytest.approx(sd)
        assert row.alpha_per_comparison == pytest.approx(alpha_pair)
        assert row.power == pytest.approx(config["planning_power"])
        assert row.n_per_arm > 0 and row.n_per_arm == int(row.n_per_arm)
        shift = row.mde_usd_per_customer / (sd * np.sqrt(2. / row.n_per_arm))
        attained = stats.norm.sf(critical - shift) + stats.norm.cdf(-critical - shift)
        assert config["planning_power"] <= attained < config["planning_power"] + .001
        assert row.n_total_3_arms == 3 * row.n_per_arm


def test_profit_unavailable_interval_is_not_reported_as_crossing_zero():
    effect = pd.DataFrame([dict(arm=ARMS[1], metric="spend", effect=.5, family_ci_low=np.nan, family_ci_high=np.nan)])
    assert set(profit_scenarios(effect, CONFIG).scenario_signal) == {"区间不可估计"}


@pytest.mark.parametrize("key,value", [("planning_mde_usd", [0]), ("margins", [1.1]), ("contact_costs_usd", [-.1]), ("bootstrap_reps", 2)])
def test_invalid_assumptions_fail(key, value):
    config = deepcopy(CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_recommendation_does_not_force_positive_result():
    assert "证据不足" in recommendation(SimpleNamespace(effect=.1, p_holm=.5), .05, False)
    assert "暂停" in recommendation(SimpleNamespace(effect=-.1, p_holm=.01), .05, False)
    assert "先排查" in recommendation(SimpleNamespace(effect=.1, p_holm=.01), .05, True)


def test_changed_cached_source_is_rejected(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "hillstrom.csv").write_text("changed", encoding="utf-8")
    (raw / "hillstrom.source.json").write_text(json.dumps({"sha256": "incorrect"}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        download(tmp_path)


def test_delivered_real_data_and_metrics():
    raw_path = ROOT / "data/raw/hillstrom.csv"
    if not raw_path.exists():
        pytest.skip("先获取公开数据，再运行真实数据核验")
    df = pd.read_csv(raw_path)
    assert len(df) == 64000
    quality = validate(df)
    assert quality["zero_spend_rows"] == 63422
    actual = sql_metrics(prepare(df), (ROOT / "sql/01_metrics.sql").read_text(encoding="utf-8"))
    reconcile(df, actual)
    assert dict(zip(actual.segment, actual.n)) == {ARMS[0]: 21306, ARMS[1]: 21307, ARMS[2]: 21387}


def test_delivered_main_results_match_independent_welch():
    path = ROOT / "reports/tables/effects.csv"
    if not path.exists():
        pytest.skip("先运行分析，再核对生成的结果")
    df = pd.read_csv(ROOT / "data/raw/hillstrom.csv")
    main = pd.read_csv(path).query("metric == 'spend'")
    pvalues = []
    for row in main.itertuples():
        treatment = df.loc[df.segment == row.arm, "spend"]
        control = df.loc[df.segment == ARMS[0], "spend"]
        p = stats.ttest_ind(treatment, control, equal_var=False).pvalue
        assert row.p_value == pytest.approx(p)
        assert row.effect == pytest.approx(treatment.mean() - control.mean())
        pvalues.append(p)
    order = np.argsort(pvalues)
    expected = np.minimum(1., np.maximum.accumulate(np.asarray(pvalues)[order] * [2, 1]))
    np.testing.assert_allclose(main.p_holm.to_numpy()[order], expected)
