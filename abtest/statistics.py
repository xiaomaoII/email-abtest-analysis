"""Explicit estimands, uncertainty and exploratory heterogeneity checks."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep

from .data import ARMS


def validate_config(config: dict) -> None:
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed 必须是非负整数")
    if not 0 < config["alpha"] < 1 or not 0 < config["srm_alpha"] < 1:
        raise ValueError("alpha 与 srm_alpha 必须位于 0 和 1 之间")
    if not isinstance(config["bootstrap_reps"], int) or config["bootstrap_reps"] < 1000:
        raise ValueError("正式分析至少需要 1000 次 bootstrap")
    if not 0 < config["planning_power"] < 1:
        raise ValueError("power 必须位于 0 和 1 之间")
    for name in ["margins", "contact_costs_usd", "planning_mde_usd"]:
        values = np.asarray(config[name], dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{name} 必须是非空有限数值列表")
    if any(x < 0 or x > 1 for x in config["margins"]):
        raise ValueError("毛利率必须位于 0—1")
    if any(x < 0 for x in config["contact_costs_usd"]):
        raise ValueError("触达成本不能为负")
    if any(x <= 0 for x in config["planning_mde_usd"]):
        raise ValueError("MDE 必须大于零")


def finite_sample(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("每组至少需要两个有限数值，不允许缺失值或空组")
    return x


def relative_lift(treatment_mean: float, control_mean: float) -> float:
    return float(treatment_mean / control_mean - 1) if control_mean != 0 else float("nan")


def welch(treatment, control, alpha=.05) -> dict:
    t, c = finite_sample(treatment), finite_sample(control)
    vt, vc = t.var(ddof=1) / len(t), c.var(ddof=1) / len(c)
    effect = float(t.mean() - c.mean())
    se = math.sqrt(vt + vc)
    if se == 0:
        return {"effect": effect, "se": 0., "df": float("nan"), "p_value": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "status": "zero_variance"}
    dof = (vt + vc) ** 2 / (vt ** 2 / (len(t) - 1) + vc ** 2 / (len(c) - 1))
    width = float(stats.t.ppf(1 - alpha / 2, dof) * se)
    return {"effect": effect, "se": se, "df": dof,
            "p_value": float(2 * stats.t.sf(abs(effect / se), dof)),
            "ci_low": effect - width, "ci_high": effect + width, "status": "ok"}


def bootstrap_difference(treatment, control, reps: int, seed: int, alpha=.05) -> tuple[float, float]:
    """Independent within-arm sampling of customers, including all zero-spend rows."""
    t, c = finite_sample(treatment), finite_sample(control)
    if reps < 2:
        raise ValueError("bootstrap reps 至少为 2")
    rng = np.random.default_rng(seed)
    effects = np.empty(reps)
    for start in range(0, reps, 64):
        batch = min(64, reps - start)
        tm = rng.choice(t, size=(batch, len(t)), replace=True).mean(axis=1)
        cm = rng.choice(c, size=(batch, len(c)), replace=True).mean(axis=1)
        effects[start:start + batch] = tm - cm
    bounds = np.quantile(effects, [alpha / 2, 1 - alpha / 2])
    return float(bounds[0]), float(bounds[1])


def experiment_effects(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    control = df[df.segment == ARMS[0]]
    rows = []
    for index, arm in enumerate(ARMS[1:]):
        treatment = df[df.segment == arm]
        for metric in ["spend", "conversion", "visit"]:
            t, c = treatment[metric].to_numpy(), control[metric].to_numpy()
            if metric == "spend":
                row = welch(t, c, config["alpha"])
                conservative = welch(t, c, config["alpha"] / 2)
                low, high = bootstrap_difference(t, c, config["bootstrap_reps"], config["seed"] + index, config["alpha"])
                row.update(bootstrap_low=low, bootstrap_high=high,
                           family_ci_low=conservative["ci_low"], family_ci_high=conservative["ci_high"],
                           method="Welch; Holm family of 2; percentile bootstrap")
            else:
                lo, hi = confint_proportions_2indep(int(t.sum()), len(t), int(c.sum()), len(c), method="newcomb", compare="diff", alpha=config["alpha"])
                test = test_proportions_2indep(int(t.sum()), len(t), int(c.sum()), len(c), method="score", compare="diff", correction=True)
                row = {"effect": float(t.mean() - c.mean()), "ci_low": float(lo), "ci_high": float(hi),
                       "p_value": float(test.pvalue), "method": "Newcombe CI; score test; descriptive secondary", "status": "ok"}
            row.update(arm=arm, metric=metric, n_treatment=len(t), n_control=len(c),
                       mean_treatment=float(t.mean()), mean_control=float(c.mean()),
                       relative_lift=relative_lift(t.mean(), c.mean()))
            rows.append(row)
    result = pd.DataFrame(rows)
    main = result.metric == "spend"
    result.loc[main, "p_holm"] = multipletests(result.loc[main, "p_value"], alpha=config["alpha"], method="holm")[1]
    return result


def randomization_checks(df: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    counts = df.segment.value_counts().reindex(ARMS)
    chi, p = stats.chisquare(counts.to_numpy(), np.full(3, len(df) / 3))
    srm = {"expected_ratio": [1/3] * 3, "counts": {k: int(v) for k, v in counts.items()},
           "chi_square": float(chi), "p_value": float(p), "alpha": config["srm_alpha"],
           "flagged": bool(p < config["srm_alpha"]),
           "interpretation": "未发现比例异常不等于证明随机化成功；实际分流机制及日志不可观测。"}
    features = df[["recency", "history", "mens", "womens", "newbie"]].astype(float).copy()
    categories = pd.get_dummies(df[["history_segment", "zip_code", "channel"]], dtype=float)
    features = pd.concat([features, categories], axis=1)
    rows = []
    for arm in ARMS[1:]:
        for col in features:
            t = features.loc[df.segment == arm, col]
            c = features.loc[df.segment == ARMS[0], col]
            denominator = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
            smd = float((t.mean() - c.mean()) / denominator) if denominator else float("nan")
            rows.append({"arm": arm, "feature": col, "treatment_mean": t.mean(),
                         "control_mean": c.mean(), "smd": smd, "abs_smd": abs(smd)})
    return srm, pd.DataFrame(rows)


def heterogeneity(df: pd.DataFrame, alpha=.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three predeclared dimensions; joint HC3 Wald tests, exploratory only."""
    interactions, effects = [], []
    for dimension in ["customer_type", "history_band", "recency_band"]:
        model = smf.ols(f"spend ~ C(segment) * C({dimension})", data=df).fit(cov_type="HC3")
        terms = [i for i, name in enumerate(model.params.index) if ":" in name]
        contrast = np.eye(len(model.params))[terms]
        test = model.wald_test(contrast, scalar=True)
        interactions.append({"dimension": dimension, "wald_chi2": float(test.statistic),
                             "df": len(terms), "p_value": float(test.pvalue), "covariance": "HC3"})
        for subgroup, subset in df.groupby(dimension, observed=True):
            for arm in ARMS[1:]:
                t = subset.loc[subset.segment == arm, "spend"]
                c = subset.loc[subset.segment == ARMS[0], "spend"]
                row = welch(t, c, alpha)
                row.update(dimension=dimension, subgroup=subgroup, arm=arm, n_treatment=len(t), n_control=len(c),
                           buyers_treatment=int((t > 0).sum()), buyers_control=int((c > 0).sum()))
                effects.append(row)
    interactions = pd.DataFrame(interactions)
    interactions["p_holm_exploratory"] = multipletests(interactions.p_value, method="holm")[1]
    return interactions, pd.DataFrame(effects)


def profit_scenarios(effects: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for r in effects[effects.metric == "spend"].itertuples():
        for margin in config["margins"]:
            for cost in config["contact_costs_usd"]:
                low = r.family_ci_low * margin - cost
                high = r.family_ci_high * margin - cost
                signal = "区间不可估计" if not np.isfinite([low, high]).all() else (
                    "区间下界为正" if low > 0 else ("区间上界为负" if high < 0 else "区间跨零或触零"))
                rows.append({"arm": r.arm, "assumed_margin": margin, "assumed_contact_cost_usd": cost,
                             "net_per_customer": r.effect * margin - cost,
                             "net_ci_low_family": low, "net_ci_high_family": high,
                             "break_even_cost_point": r.effect * margin,
                             "break_even_cost_lower_bound": r.family_ci_low * margin,
                             "scenario_signal": signal})
    return pd.DataFrame(rows)


def sample_size_plan(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Normal approximation for a future three-arm design, not post-hoc power."""
    historical_sd = float(np.sqrt(metrics.spend_variance.max()))
    z_alpha = stats.norm.ppf(1 - config["alpha"] / (2 * 2))
    z_power = stats.norm.ppf(config["planning_power"])
    rows = []
    for sd_multiplier in [1., 1.25]:
        for mde in config["planning_mde_usd"]:
            sd = historical_sd * sd_multiplier
            n = math.ceil(2 * sd ** 2 * (z_alpha + z_power) ** 2 / mde ** 2)
            rows.append({"mde_usd_per_customer": mde, "power": config["planning_power"],
                         "alpha_per_comparison": config["alpha"] / 2,
                         "historical_sd_usd": historical_sd, "sd_multiplier": sd_multiplier,
                         "assumed_sd_usd": sd, "n_per_arm": n, "n_total_3_arms": n * 3})
    return pd.DataFrame(rows)


def recommendation(row, alpha: float, srm_flagged: bool) -> str:
    if srm_flagged:
        return "先排查分流比例异常，暂缓因果推广建议"
    if not np.isfinite(row.p_holm):
        return "统计推断不可用，先核实数据与方法"
    if row.p_holm < alpha and row.effect > 0:
        return "收入增量有统计证据；在成本和护栏满足条件后开展当前业务验证"
    if row.p_holm < alpha and row.effect < 0:
        return "收入下降有统计证据；暂停推广并排查原因"
    return "收入证据不足；保留不确定性，按业务阈值设计下一轮实验"
