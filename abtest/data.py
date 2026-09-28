"""Download and validate the source; never infer user identity from equal attributes."""
from __future__ import annotations

import hashlib
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

SOURCE_PAGE = "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
SOURCE_CSV = "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
MIRROR_CSV = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
MIRROR_DOC = "https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html#fetch_hillstrom"
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
LABELS = dict(zip(ARMS, ["不发邮件", "男装邮件", "女装邮件"]))
COLUMNS = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend"]


def download(root: Path) -> Path:
    path = root / "data/raw/hillstrom.csv"
    provenance = path.with_suffix(".source.json")
    if path.exists():
        if not provenance.exists():
            raise ValueError("原始数据缺少来源记录，请恢复 hillstrom.source.json 或移走 CSV 后重新获取。")
        recorded = json.loads(provenance.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]:
            raise ValueError("原始 CSV 与来源记录的 SHA256 不符；请检查本地文件，不会自动覆盖。")
        return path
    fallback_reason = None
    requested_url = SOURCE_CSV
    def fetch(url):
        request = Request(url, headers={"User-Agent": "Hillstrom-Portfolio/1.0"})
        with urlopen(request, timeout=45) as response:
            return response.read(), response.url
    try:
        payload, resolved_url = fetch(SOURCE_CSV)
    except (URLError, TimeoutError) as error:
        # A blocked socket must be escalated, not silently retried via another host.
        if getattr(getattr(error, "reason", error), "winerror", None) == 10013:
            raise
        fallback_reason = str(error)
        requested_url = MIRROR_CSV
        compressed, resolved_url = fetch(MIRROR_CSV)
        if hashlib.md5(compressed).hexdigest() != "a68a81291f53a14f4e29002629803ba3":
            raise ValueError("scikit-uplift 镜像压缩文件与公开 MD5 不符")
        payload = gzip.decompress(compressed)
    df = pd.read_csv(io.BytesIO(payload))
    validate(df)
    if len(df) != 64000:
        raise ValueError(f"下载结果不是预期的 64000 行：{len(df)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".part")
    staging.write_bytes(payload)
    staging.replace(path)
    provenance.write_text(json.dumps({
        "source_page": SOURCE_PAGE, "requested_url": requested_url,
        "fallback_reason": fallback_reason, "mirror_documentation": MIRROR_DOC if fallback_reason else None,
        "resolved_url": resolved_url, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "rows": len(df), "experiment_year": 2008, "outcome_window_days": 14,
        "license_note": "作者公开用于分析；原页面未提供明确的标准许可，不将原始数据默认纳入版本控制。"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate(df: pd.DataFrame) -> dict:
    missing = sorted(set(COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    if df.empty:
        raise ValueError("数据为空")
    nulls = df[COLUMNS].isna().sum()
    if nulls.any():
        raise ValueError(f"存在缺失值，需核实后再分析，不能自动填零：{nulls[nulls > 0].to_dict()}")
    numeric = ["recency", "history", "mens", "womens", "newbie", "visit", "conversion", "spend"]
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col]) or not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} 必须为有限数值")
    for col in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} 必须为 0/1")
    if not (df.recency.between(1, 12) & (df.recency % 1 == 0)).all():
        raise ValueError("recency 必须为 1—12 的整数")
    if (df[["history", "spend"]] < 0).any().any():
        raise ValueError("消费金额不能为负")
    if set(df.segment) != set(ARMS):
        raise ValueError("必须恰好包含三个实验组，不能缺组或包含未知组")
    if (df.segment.value_counts() < 2).any():
        raise ValueError("每组至少需要两个样本")
    allowed = {
        "channel": {"Phone", "Web", "Multichannel"},
        "zip_code": {"Urban", "Suburban", "Surburban", "Rural"},
        "history_segment": {"1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"},
    }
    for col, values in allowed.items():
        if not set(df[col]).issubset(values):
            raise ValueError(f"{col} 包含未知类别：{set(df[col]) - values}")
    contradictions = {
        "conversion_without_visit": int(((df.conversion == 1) & (df.visit == 0)).sum()),
        "spend_without_conversion": int(((df.spend > 0) & (df.conversion == 0)).sum()),
        "conversion_without_spend": int(((df.conversion == 1) & (df.spend == 0)).sum()),
    }
    if any(contradictions.values()):
        raise ValueError(f"结果字段存在矛盾，先核实来源：{contradictions}")
    return {
        "rows": len(df), "missing_by_column": {c: int(v) for c, v in nulls.items()},
        "contradictions": contradictions,
        "identical_rows_after_first": int(df[COLUMNS].duplicated().sum()),
        "identical_rows_action": "保留：无原始用户 ID，相同属性不代表同一人。",
        "zero_spend_rows": int((df.spend == 0).sum()),
        "spend_max": float(df.spend.max()),
        "spend_quantiles": {str(q): float(df.spend.quantile(q)) for q in [.5, .95, .99, .999]},
        "amount_unit": "USD", "exclusions": 0,
        "source_spelling_surburban_rows": int((df.zip_code == "Surburban").sum()),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zip_code"] = out.zip_code.replace({"Surburban": "Suburban"})
    out.insert(0, "source_row_id", np.arange(1, len(out) + 1))
    out["customer_type"] = out.newbie.map({0: "已有客户", 1: "近12月新客户"})
    out["history_band"] = pd.cut(out.history, [-np.inf, 100, 350, np.inf], labels=["≤100美元", "100–350美元", ">350美元"]).astype(str)
    out["recency_band"] = pd.cut(out.recency, [0, 3, 6, 12], labels=["1–3个月", "4–6个月", "7–12个月"]).astype(str)
    return out


def sql_metrics(df: pd.DataFrame, sql: str, database: Path | None = None) -> pd.DataFrame:
    import duckdb

    connection = duckdb.connect(str(database) if database else ":memory:")
    try:
        connection.register("source_data", df)
        connection.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM source_data")
        return connection.execute(sql).df()
    finally:
        connection.close()


def reconcile(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    independent = df.groupby("segment").agg(
        n=("spend", "size"), visitors=("visit", "sum"), buyers=("conversion", "sum"),
        revenue=("spend", "sum"), revenue_per_customer=("spend", "mean"),
        visit_rate=("visit", "mean"), conversion_rate=("conversion", "mean"),
        spend_variance=("spend", "var"),
    ).sort_index()
    actual = metrics.set_index("segment").sort_index()[independent.columns]
    if not actual.index.equals(independent.index):
        raise AssertionError("SQL/Python 分组不一致")
    np.testing.assert_allclose(actual.to_numpy(dtype=float), independent.to_numpy(dtype=float), rtol=1e-10, atol=1e-8)
