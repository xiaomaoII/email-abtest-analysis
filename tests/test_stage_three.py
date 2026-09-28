"""Independent checks for exploratory contrasts, cost decisions and saved exercises."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from abtest.data import ARMS, prepare
from abtest.statistics import heterogeneity, profit_scenarios


def test_grouping_uses_only_pre_experiment_attributes_and_fixed_boundaries():
    raw = pd.DataFrame({
        "newbie": [0, 1, 0, 1, 0, 1], "zip_code": ["Urban"] * 6,
        "history": [0, 100, 100.01, 350, 350.01, 1000],
        "recency": [1, 3, 4, 6, 7, 12],
        "visit": [0] * 6, "conversion": [0] * 6, "spend": [0.] * 6,
    })
    before = prepare(raw)
    raw[["visit", "conversion", "spend"]] = [1, 1, 100.]
    after = prepare(raw)
    columns = ["customer_type", "history_band", "recency_band"]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert before.history_band.tolist() == ["≤100美元", "≤100美元", "100–350美元", "100–350美元", ">350美元", ">350美元"]
    assert before.recency_band.tolist() == ["1–3个月"] * 2 + ["4–6个月"] * 2 + ["7–12个月"] * 2


def test_joint_hc3_wald_matches_independent_cell_mean_contrasts():
    # Unbalanced, heteroskedastic cells make HC0/HC1/HC2 and omitted contrasts detectable.
    rows = []
    for group in range(3):
        for arm_index, arm in enumerate(ARMS):
            n = 5 + group + arm_index
            for i in range(n):
                rows.append({"segment": arm,
                             "customer_type": str(group), "history_band": str(group),
                             "recency_band": str(group),
                             "spend": 0. if i < 2 else float((i + 1) * (1 + group * arm_index))})
    df = pd.DataFrame(rows)
    actual, subgroups = heterogeneity(df)
    cells = df.groupby(["customer_type", "segment"]).spend.agg(["mean", "var", "size"])
    keys = list(cells.index)
    contrasts = []
    for level in ["1", "2"]:
        for arm in ARMS[1:]:
            row = np.zeros(len(keys))
            for key, sign in [((level, arm), 1), ((level, ARMS[0]), -1),
                              (("0", arm), -1), (("0", ARMS[0]), 1)]:
                row[keys.index(key)] = sign
            contrasts.append(row)
    r = np.array(contrasts)
    # A saturated cell-mean regression has h_ii=1/n_g, hence HC3 var(mean_g)=s_g²/(n_g-1).
    cov = r @ np.diag(cells["var"] / (cells["size"] - 1)) @ r.T
    delta = r @ cells["mean"].to_numpy()
    wald = float(delta @ np.linalg.solve(cov, delta))
    p = stats.chi2.sf(wald, 4)
    np.testing.assert_allclose(actual.wald_chi2, wald, rtol=1e-10)
    np.testing.assert_allclose(actual.p_value, p, rtol=1e-10)
    np.testing.assert_allclose(actual.p_holm_exploratory, min(1., 3 * p))
    assert (actual["df"] == 4).all()
    for row in subgroups.itertuples():
        t = df[(df[row.dimension] == row.subgroup) & (df.segment == row.arm)].spend
        c = df[(df[row.dimension] == row.subgroup) & (df.segment == ARMS[0])].spend
        reference = stats.ttest_ind(t, c, equal_var=False)
        np.testing.assert_allclose([row.p_value, row.ci_low, row.ci_high],
                                  [reference.pvalue, *reference.confidence_interval()], rtol=1e-10)
        assert row.n_treatment == len(t) and row.n_control == len(c)


def test_profit_decisions_include_interval_endpoints_and_zero_margin():
    effects = pd.DataFrame([dict(arm=ARMS[2], metric="spend", effect=.5,
                                 family_ci_low=.25, family_ci_high=.75)])
    costs = [0., .125, .25, .375, .5]
    result = profit_scenarios(effects, {"margins": [.5, 0.], "contact_costs_usd": costs})
    positive_margin = result[result.assumed_margin == .5]
    assert positive_margin.scenario_signal.tolist() == [
        "区间下界为正", "区间跨零或触零", "区间跨零或触零", "区间跨零或触零", "区间上界为负"]
    np.testing.assert_allclose(positive_margin.net_per_customer, .25 - np.array(costs))
    np.testing.assert_allclose(positive_margin.break_even_cost_point, .25)
    np.testing.assert_allclose(positive_margin.break_even_cost_lower_bound, .125)
    zero_margin = result[result.assumed_margin == 0.]
    for column in ["net_per_customer", "net_ci_low_family", "net_ci_high_family"]:
        np.testing.assert_allclose(zero_margin[column], -np.array(costs))


def test_rebuilding_keeps_stage_three_practice_and_executes_it(tmp_path, monkeypatch):
    import nbformat
    from abtest.notebook import build_notebook

    for variable in ["JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "IPYTHONDIR"]:
        monkeypatch.delenv(variable, raising=False)
    ids = ["992ae00247d4e0af", "7c5d9288a25c4670", "98388394faf70936", "3998930164f06feb"]
    df = pd.DataFrame([
        {"customer_type": group, "segment": arm, "spend": spend}
        for group, amounts in [("已有客户", [2., 4., 5.]), ("近12月新客户", [1., 6., 3.])]
        for arm, amount in zip(ARMS, amounts) for spend in [0., amount]
    ])
    effects = pd.DataFrame([dict(arm=arm, metric="spend", effect=.5,
                                family_ci_low=.25, family_ci_high=.75) for arm in ARMS[1:]])
    config = {"margins": [.2, .4], "contact_costs_usd": [0., .1, .2]}
    for _ in range(2):
        notebook = nbformat.read(build_notebook(tmp_path, execute=False), as_version=4)
        cells = [c for c in notebook.cells if c.id in ids]
        assert [c.id for c in cells] == ids
        assert cells[1].cell_type == "markdown"
        display = []
        namespace = dict(df=df, effects=effects, config=config,
                         profit_scenarios=profit_scenarios, display=display.append)
        for cell in cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, cell.id, "exec"), namespace)
        assert len(display) == 3
        np.testing.assert_allclose(namespace["effect_difference"].to_numpy(), [1.5, -.5])
        assert len(namespace["women_40"]) == 3
        np.testing.assert_allclose(namespace["women_40"].net_per_customer, [.2, .1, 0.])
