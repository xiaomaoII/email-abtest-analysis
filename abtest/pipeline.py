"""One reproducible workflow; reports consume these output tables."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import download, prepare, reconcile, sql_metrics, validate
from .statistics import (experiment_effects, heterogeneity, profit_scenarios,
                         randomization_checks, sample_size_plan, validate_config)


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def pipeline(root: Path, prepare_only=False) -> dict:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    for directory in ["data/processed", "reports/tables", "reports/figures", "docs", "notebooks"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    tables = root / "reports/tables"
    raw_path = download(root)
    raw = pd.read_csv(raw_path)
    quality = validate(raw)
    df = prepare(raw)
    df.to_csv(root / "data/processed/customers.csv", index=False, encoding="utf-8-sig")
    database = root / "data/processed/hillstrom.duckdb"
    metrics = sql_metrics(df, (root / "sql/01_metrics.sql").read_text(encoding="utf-8"), database)
    reconcile(df, metrics)
    segments = sql_metrics(df, (root / "sql/02_segments.sql").read_text(encoding="utf-8"))
    metrics.to_csv(tables / "group_metrics.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(tables / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    quality["sql_python_reconciliation"] = "passed"
    save_json(tables / "quality.json", quality)
    print(f"数据核验与 SQL/Python 对账完成：{len(df):,} 行。", flush=True)
    if prepare_only:
        return {"data": df, "quality": quality, "metrics": metrics}
    srm, balance = randomization_checks(df, config)
    print("计算收入与辅助指标效果、用户层级 Bootstrap……", flush=True)
    effects = experiment_effects(df, config)
    interactions, subgroups = heterogeneity(df, config["alpha"])
    scenarios = profit_scenarios(effects, config)
    planning = sample_size_plan(metrics, config)
    results = {"effects": effects, "baseline_balance": balance, "interaction_tests": interactions,
               "subgroup_effects": subgroups, "profit_scenarios": scenarios, "sample_size_plan": planning}
    for name, table in results.items():
        table.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig", na_rep="")
    save_json(tables / "randomization.json", srm)
    from .reporting import plot_results, write_reports
    plot_results(root, effects, balance, subgroups, config)
    write_reports(root, metrics, effects, quality, srm, balance, interactions, scenarios, planning, config)
    tracked_inputs = [root / "config.json", root / "requirements.txt", root / "run.py"]
    if (root / "requirements-lock.txt").exists():
        tracked_inputs.append(root / "requirements-lock.txt")
    tracked_inputs += sorted((root / "abtest").glob("*.py")) + sorted((root / "sql").glob("*.sql"))
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "statsmodels", "duckdb", "matplotlib", "nbformat", "nbclient", "ipykernel", "pytest"]}
    save_json(root / "reports/run_manifest.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "config": config, "packages": versions,
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "inputs_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_inputs},
        "table_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tables.glob("*"))},
        "intervals": {"pointwise": 1-config["alpha"], "family": 1-config["alpha"],
                      "bonferroni_individual": 1-config["alpha"]/2},
        "note": "表中的空字段代表不适用或不可用，绝不等于零；金额均为美元。"
    })
    print("分析表、三张图、中文决策报告与面试材料已生成。", flush=True)
    return {"data": df, "quality": quality, "metrics": metrics, **results}
