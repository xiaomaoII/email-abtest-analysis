"""Generate human-readable outputs from the same numeric result tables."""
from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .data import ARMS, LABELS, SOURCE_PAGE
from .statistics import recommendation

DIMENSIONS = {"customer_type": "新老客户", "history_band": "历史消费水平", "recency_band": "距上次购买时间"}


def baseline_label(feature: str) -> str:
    fixed = {"recency": "距上次购买（月）", "history": "过去一年消费金额", "mens": "曾购买男装",
             "womens": "曾购买女装", "newbie": "近12月新客户", "channel_Phone": "渠道：电话",
             "channel_Web": "渠道：网站", "channel_Multichannel": "渠道：多渠道",
             "zip_code_Urban": "区域：城市", "zip_code_Suburban": "区域：郊区", "zip_code_Rural": "区域：乡村"}
    if feature.startswith("history_segment_"):
        band = feature.removeprefix("history_segment_").split(") ", 1)[-1].replace("$", "")
        return f"历史消费：{band} 美元"
    return fixed.get(feature, feature)


def number(value, digits=4) -> str:
    if not np.isfinite(value):
        return "不可用"
    if 0 < abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:,.{digits}f}"


def md_table(headers, rows) -> str:
    def clean(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def plot_results(root: Path, effects: pd.DataFrame, balance: pd.DataFrame, subgroups: pd.DataFrame, config: dict) -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((x for x in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"] if x in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#F8FAFC", "axes.facecolor": "#F8FAFC"})
    dest = root / "reports/figures"
    dest.mkdir(parents=True, exist_ok=True)
    main = effects[effects.metric == "spend"].set_index("arm").loc[ARMS[1:]]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    y = np.arange(2)
    for offset, lo, hi, label, color in [
        (-.15, "ci_low", "ci_high", f"Welch {(1-config['alpha']):.0%} 单项区间", "#225EA8"),
        (0, "bootstrap_low", "bootstrap_high", f"Bootstrap {(1-config['alpha']):.0%} 单项区间", "#D97706"),
        (.15, "family_ci_low", "family_ci_high", f"Bonferroni {(1-config['alpha']):.0%} 家族区间", "#14857B"),
    ]:
        ax.errorbar(main.effect, y + offset, xerr=[main.effect - main[lo], main[hi] - main.effect],
                    fmt="o", capsize=4, color=color, label=label, markersize=5)
    ax.axvline(0, color="#64748B", linestyle="--", linewidth=1)
    ax.set_yticks(y, [LABELS[x] for x in main.index])
    ax.set_xlabel("相对不发邮件：两周人均收入增量（美元 / 随机分组客户）")
    ax.set_title("邮件带来的收入增量与不确定性", loc="left", fontsize=16, pad=16)
    ax.grid(axis="x", alpha=.18)
    ax.set_ylim(-.6, 1.8)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.savefig(dest / "revenue_effects.png", dpi=180)
    plt.close(fig)

    largest = balance.groupby("feature").abs_smd.max().nlargest(12).index
    fig, ax = plt.subplots(figsize=(9.5, 6), layout="constrained")
    for arm, color in zip(ARMS[1:], ["#225EA8", "#D97706"]):
        data = balance[balance.arm == arm].set_index("feature").loc[largest]
        ax.scatter(data.abs_smd, np.arange(len(largest)), label=LABELS[arm], color=color, s=35)
    ax.axvline(.1, linestyle="--", color="#64748B", label="0.1 经验参考线")
    ax.set_yticks(np.arange(len(largest)), [baseline_label(x) for x in largest])
    ax.invert_yaxis()
    ax.set_xlabel("相对对照组的绝对标准化均值差（SMD）")
    ax.set_title("实验前属性平衡诊断：偏差最大的 12 项", loc="left", fontsize=15)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.18)
    fig.savefig(dest / "baseline_balance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained", sharex=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        subset = subgroups[subgroups.dimension == dimension]
        groups = {"customer_type": ["已有客户", "近12月新客户"],
                  "history_band": ["≤100美元", "100–350美元", ">350美元"],
                  "recency_band": ["1–3个月", "4–6个月", "7–12个月"]}[dimension]
        for arm, offset, color in zip(ARMS[1:], [-.12, .12], ["#225EA8", "#D97706"]):
            data = subset[subset.arm == arm].set_index("subgroup").loc[groups]
            ax.errorbar(data.effect, np.arange(len(groups)) + offset,
                        xerr=[data.effect - data.ci_low, data.ci_high - data.effect], fmt="o", capsize=3,
                        color=color, label=LABELS[arm], markersize=4)
        ax.set_yticks(np.arange(len(groups)), groups)
        ax.invert_yaxis()
        ax.set_title(DIMENSIONS[dimension], loc="left")
        ax.set_xlabel("两周人均收入增量（美元）")
        ax.axvline(0, color="#64748B", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=.18)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"探索性分群：{(1-config['alpha']):.0%} 单项区间，尚未验证定向策略", fontsize=16)
    fig.savefig(dest / "subgroup_effects.png", dpi=180)
    plt.close(fig)


def write_reports(root: Path, metrics: pd.DataFrame, effects: pd.DataFrame, quality: dict,
                  srm: dict, balance: pd.DataFrame, interactions: pd.DataFrame,
                  scenarios: pd.DataFrame, planning: pd.DataFrame, config: dict) -> None:
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    main = effects[effects.metric == "spend"].copy()
    interaction_signal = bool((interactions.p_holm_exploratory < config["alpha"]).any())
    summary_subgroups = "分群交互检验存在探索性信号，仍需新样本验证。" if interaction_signal else "三个预设分群维度的交互检验经校正后均未给出充分差异证据，不据本次结果推荐定向投放。"
    interval_pct = f"{1-config['alpha']:.0%}"
    family_individual_pct = f"{1-config['alpha']/2:.1%}"
    group_table = md_table(["分组", "客户数", "访问率", "购买率", "两周人均收入（美元）"], [
        [LABELS[r.segment], f"{r.n:,}", f"{r.visit_rate:.2%}", f"{r.conversion_rate:.2%}", number(r.revenue_per_customer)]
        for r in metrics.itertuples()])
    effect_table = md_table(["相对不发邮件", "人均收入增量（美元）", "相对提升", f"Welch {interval_pct} 区间", "Holm校正p值"], [
        [LABELS[r.arm], number(r.effect), f"{r.relative_lift:.1%}", f"[{number(r.ci_low)}, {number(r.ci_high)}]", number(r.p_holm, 6)]
        for r in main.itertuples()])
    decisions = "\n".join(f"- **{LABELS[r.arm]}：**{recommendation(r, config['alpha'], srm['flagged'])}。" for r in main.itertuples())
    summary = f"""# 营销触达的增量收益：业务决策摘要

**项目性质：**2008 年公开随机实验的独立复盘；并非本人实施的线上实验。金额单位为美元，结果窗口为两周。

## 当前证据支持什么

{decisions}

两种邮件分别与对照组比较；本项目未做邮件之间的确认性直接比较，因此不宣布哪种邮件胜出。

{summary_subgroups}

{effect_table}

主指标为全部随机分组客户的人均收入。上表区间为单项 {interval_pct} 区间，显著性结论以两项收入检验的 Holm 校正 p 值为准。

## 推广需要满足的条件

- 先拿到实际毛利率和每名分组客户的触达成本，按“收入增量 × 毛利率 − 触达成本”评估。数据没有实际成本，不能声称已经盈利。
- 情景分析采用两项比较的 Bonferroni 家族区间；只有净收益下界为正时才说明该假设下证据较稳健，仍需在当前业务验证。
- 当前没有退订、投诉、长期留存或送达日志；下一轮补齐这些指标，并明确可接受阈值。
- 分群结果只用于提出后续实验假设，不能直接得出精准投放已经有效。

## 可信度与下一步

共 {quality['rows']:,} 行，分析排除 {quality['exclusions']} 行；分流比例检验 p={number(srm['p_value'], 6)}，最大绝对 SMD={number(balance.abs_smd.max())}。{'分流比例触发异常，须先排查。' if srm['flagged'] else '未发现明显分流比例异常；这不等于证明随机化成功。'}

下一步：确认业务收益门槛，采用用户级 1∶1∶1 随机分组，在新样本中验证收入及探索性人群假设。样本量测算见 [完整报告](analysis_report.md)，学习路线见 [分阶段练习](../docs/learning_path.md)。

数字来源：`tables/group_metrics.csv`、`tables/effects.csv`、`tables/baseline_balance.csv`、`tables/quality.json`、`tables/randomization.json`。数据出处：[Kevin Hillstrom 原始说明]({SOURCE_PAGE})。
"""
    (reports / "decision_summary.md").write_text(summary, encoding="utf-8")
    bootstrap_table = md_table(["邮件", f"Bootstrap {interval_pct}", f"Bonferroni单项{family_individual_pct}"], [
        [LABELS[r.arm], f"[{number(r.bootstrap_low)}, {number(r.bootstrap_high)}]", f"[{number(r.family_ci_low)}, {number(r.family_ci_high)}]"]
        for r in main.itertuples()])
    auxiliary = effects[effects.metric != "spend"]
    secondary_table = md_table(["邮件", "辅助指标", "增量（百分点）", f"{interval_pct}区间（百分点）", "未校正p值"], [
        [LABELS[r.arm], {"conversion": "购买率", "visit": "访问率"}[r.metric], number(r.effect*100, 3),
         f"[{number(r.ci_low*100,3)}, {number(r.ci_high*100,3)}]", number(r.p_value, 6)] for r in auxiliary.itertuples()])
    interaction_table = md_table(["分群维度", "联合交互检验p值", "三项Holm校正p值"], [
        [DIMENSIONS[r.dimension], number(r.p_value, 6), number(r.p_holm_exploratory, 6)] for r in interactions.itertuples()])
    candidate_dimensions = [DIMENSIONS[r.dimension] for r in interactions.itertuples() if r.p_holm_exploratory < config["alpha"]]
    subgroup_note = ("存在探索性交互信号的维度：" + "、".join(candidate_dimensions) + "。须由新样本验证，不能据此认定某个细分人群有效。") if candidate_dimensions else "三项校正后的交互检验均未给出充分证据；不据点估计排序挑选所谓最优人群。"
    scenario_table = md_table(["邮件", "假设毛利率", "点估计可承受成本", "家族区间下界可承受成本"], [
        [LABELS[r.arm], f"{r.assumed_margin:.0%}", number(r.break_even_cost_point), number(r.break_even_cost_lower_bound)]
        for r in scenarios.drop_duplicates(["arm", "assumed_margin"]).itertuples()])
    power_table = md_table(["人均收入MDE（美元）", "标准差倍数", "每组客户数", "三组合计"], [
        [r.mde_usd_per_customer, r.sd_multiplier, f"{r.n_per_arm:,}", f"{r.n_total_3_arms:,}"] for r in planning.itertuples()])
    report = f"""# 营销触达的增量收益评估与人群运营建议

## 1. 业务问题与分析约定

目标是估计邮件营销相对自然购买带来的额外收入，为后续投放实验提供依据。分析以公开数据为基础，未执行真实业务投放。

以随机分组为分析口径（ITT 原则），保留所有客户。邮件主题是男装或女装，不能据此推断客户性别。主指标为两周总消费金额 / 分组客户数；访问率和购买率均按全部客户计算。

本项目在计算结果前将收入指定为主指标，但这是历史数据复盘，不能称为原实验事前注册。未获得原实验实施日志，因果解释依赖作者描述的随机分组、组间无干扰及完整观察等假设。

## 2. 数据质量与指标

来源：[原作者数据说明]({SOURCE_PAGE})。原始文件来源、下载时间与 SHA256 记录在 `data/raw/hillstrom.source.json`，运行参数记录在 `reports/run_manifest.json`。

共 {quality['rows']:,} 行，关键字段无缺失，结果字段矛盾为零，排除 {quality['exclusions']} 行。零消费记录 {quality['zero_spend_rows']:,} 行全部保留；最高消费 {quality['spend_max']:,.2f} 美元，未经业务依据不删除。完全相同字段记录的后续出现次数为 {quality['identical_rows_after_first']:,}，因无原始用户 ID 全部保留。`source_row_id` 仅为文件行号。

{group_table}

SQL 与 pandas 分别汇总，客户数、收入、转化、访问与收入方差通过数值一致性检查。数据审计程序遇到缺失、未知类别、空组或矛盾记录会停止，不静默修补。

## 3. 实验可信度诊断

预期三组各占 1/3；卡方分流比例检验 p={number(srm['p_value'],6)}，异常阈值 {config['srm_alpha']}。{'当前触发 SRM，暂缓因果结论。' if srm['flagged'] else '当前未触发 SRM。'}

对实验前连续属性、二元属性及类别独热变量计算相对对照组的标准化均值差，最大绝对 SMD={number(balance.abs_smd.max())}。图中 0.1 是经验参考值，不是随机化的证明。平衡结果不用于事后删人或调整分流。

![实验前属性平衡诊断](figures/baseline_balance.png)

## 4. 总体实验效果

{effect_table}

![两周人均收入增量](figures/revenue_effects.png)

两个收入比较构成主要检验家族：双侧 Welch 检验、家族显著性水平 {config['alpha']}、Holm 校正。常规区间为单项 {interval_pct}；另提供 Bonferroni 单项 {family_individual_pct} 区间，使两个比较的同时覆盖率至少为 {interval_pct}（基于 Welch 近似）。它更保守，不等同于 Holm 校正区间。

{bootstrap_table}

Bootstrap 在各组内以客户为单位有放回抽样，包含所有零消费，重复 {config['bootstrap_reps']:,} 次，固定种子 {config['seed']}。收入偏态和稀疏购买下，以百分位区间作为稳健性对照；它不能消除数据缺陷或补回缺失的长期结果。相对提升仅作为描述性点估计，未估计其置信区间；对照均值为零时输出不可用。

{secondary_table}

辅助指标使用 Newcombe 比例差区间与 score 检验，未做确认性多重校正，仅用于机制解释。访问是实验后变量，不能筛出访问客户后再把购买率差解释为总体因果效果。没有订单数，不能把每位购买客户消费额称为客单价。

## 5. 探索性人群分析

事先固定三个维度：新老客户、历史消费 ≤100 / 100–350 / >350 美元、距上次购买 1–3 / 4–6 / 7–12 个月。分界是项目分析约定，不是学习得到的最优切点。

每个维度单独拟合 `spend ~ C(segment) * C(维度)`，以 HC3 稳健协方差对所有交互项做联合 Wald 检验；三个探索性检验再用 Holm 校正。

{interaction_table}

{subgroup_note}

![探索性分群效果](figures/subgroup_effects.png)

分群图为未经多重校正的单项区间。检验的是“组间效果差异”，不能用某群显著、另一群不显著代替交互检验。联合显著也不能定位具体人群；样本细分后购买事件减少，估计可能不稳定。详表同时提供分组人数与购买人数，不据本次样本计算已验证的精准投放 ROI。

## 6. 收益与推广建议

净收益增量 / 分组客户 = 人均收入增量 × 假设毛利率 − 假设触达成本。默认毛利率情景为 {', '.join(f'{x:.0%}' for x in config['margins'])}，触达成本情景为 {config['contact_costs_usd']} 美元。所有参数均为教学假设，可在 `config.json` 修改。

{scenario_table}

上述成本单位为美元 / 客户。下界为负表示无法证明任何非负成本下净收益为正，不代表实际存在负成本。区间仅反映收入估计的不确定性，不包含毛利率、送达率和成本的不确定性，也未覆盖固定实施费用或长期效应。

{decisions}

统计证据只支持该历史样本。当前业务应先核实单位经济性，再用新样本开展小范围验证；分流异常时必须先排查。

## 7. 下一轮实验设计与样本量

用户级 1∶1∶1 分为不发邮件、男装邮件、女装邮件；分组持久化，按初始分组分析，避免跨组重复触达。实验前确定合格客群、排除规则、收入定义、14 天成熟窗口、可接受的投诉与退订阈值。未成熟客户等待完整窗口，退款计入方式预先锁定。

下表为未来实验设计练习，不能称为原实验事前功效或用来解释原实验 p 值。采用各历史组中最大的收入标准差 {planning.historical_sd_usd.iloc[0]:.4f} 美元，再做 1.25 倍标准差敏感性分析。采用正态近似，双侧 Bonferroni 单比较 alpha={config['alpha']/2}，目标功效 {config['planning_power']:.0%}。

每组样本量 = ceil[2 × σ² × (z_(1−α/(2×2)) + z_power)² / MDE²]。三组等额分流，总量为每组的三倍；功效针对单个预设比较，不保证两项同时检出。MDE 是业务值得识别的最小绝对收入增量，不从当前显著结果倒推。

{power_table}

下一轮需用近期先导数据更新方差，并验证收入重尾下的功效近似。没有每日合格流量，无法给出真实运行天数；总历时需要覆盖招募时间及最后一批用户的 14 天窗口。固定样本设计下不按每日 p 值提前停止。安全护栏单独监控，触发预设安全规则可暂停，但必须说明停止原因。

若验证人群策略，先在新实验中确认交互假设；再将定向策略与全量策略随机比较并评估同一合格客群的人均净收益，避免用本样本筛人又用本样本评估策略。

## 8. 复现与适用边界

运行方式见 [README](../README.md)，完整数值见 `tables/`，逐步计算见 [分析笔记本](../notebooks/analysis.ipynb)。学习与面试材料见 `docs/`。

没有用户 ID、日级日志、邮件打开与送达日志、实验前同窗收入、成本、毛利率和长期护栏数据；不能做用户跨组排重、时序效应、真实利润或长期留存结论。不将过去一年消费额直接冒充 CUPED 的实验前同窗指标。公开数据年龄限制外推，但不影响练习实验推断和业务决策过程。
"""
    (reports / "analysis_report.md").write_text(report, encoding="utf-8")
    write_interview(root, main, interactions, config)


def write_interview(root, main, interactions, config):
    findings = "；".join(f"{LABELS[r.arm]}相对不发邮件的人均收入增量为 {r.effect:.4f} 美元，Holm 校正 p={number(r.p_holm, 6)}" for r in main.itertuples())
    text = f"""# 简历与面试表达

## 使用前提

以下是项目表达参考。只有在你亲自复现、完成练习并能解释关键判断后，才采用“独立完成”等措辞。使用 AI 辅助的情况如实说明；不要将公开数据复盘描述为任职经历或本人上线实验。

## 简历项目描述

**营销触达增量收益评估｜公开数据分析项目｜SQL、Python、DuckDB**

- 基于 6.4 万客户的三组随机实验，构建用户级指标口径，通过 SQL / Python 双重汇总核对，完成缺失、分流比例和实验前属性平衡诊断。
- 以两周人均收入为主指标，使用 Welch 检验、{config['bootstrap_reps']:,} 次用户层级 Bootstrap 与 Holm 校正评估营销增量；{findings}。
- 结合三类客户属性交互检验、毛利与成本情景和后续实验样本量测算，形成有证据边界的推广建议；未将历史结果描述为实际上线收益。

## 约三分钟介绍参考

我做的是一个公开邮件营销实验的复盘。业务问题是：如果客户原本就会购买，发邮件究竟带来了多少额外收入，是否值得继续投入？数据包含约 6.4 万名客户，随机进入两种营销邮件组和不发邮件组，观察两周结果。

我先明确了指标和分母。主指标是全部分组客户的人均收入，而不是只看购买者或访问者。因为访问和购买都可能受到邮件影响，先筛选这些人会破坏原来的随机比较。SQL 负责指标汇总，Python 独立计算并核对，也检查了缺失、字段矛盾、分流比例和实验前属性平衡。

统计分析中，收入有大量零值和少量高消费。我保留这些客户，使用允许组间方差不同的 Welch 检验，再用用户层级 Bootstrap 检查区间。由于同时比较两种邮件与对照，我对两项主要检验做 Holm 校正。实际结果是：{findings}。我会结合报告里的区间说明结果有多不确定，也不会把不显著说成完全没效果。

业务建议上，我把收入增量和盈利分开。公开数据没有成本和毛利，所以只测算假设下的盈亏平衡条件。新老客户、消费历史和购买间隔的分群，也通过交互检验判断差异；这些仍是探索，需要用新样本验证。我没有据此声称精准投放已提升收益。

这个项目最主要的产出是把证据转成下一步决策：哪些方案值得在当前业务继续验证，需要什么成本信息，以及如何安排下一轮实验。我同时明确了历史数据、两周窗口和缺失护栏对结论的限制。

## 面试追问与回答要点

1. **为什么一定要有对照组？** 对照组估计同类客户在不发邮件时的自然购买；收入总额不能直接归因于营销。
2. **为什么保留零消费？** 目标是平均每名合格分组客户产生的增量收入，零值属于该结果分布。
3. **为什么不只比较购买者消费额？** 购买是实验后变量，不同策略可能改变购买者构成；且没有订单数，不能称为客单价。
4. **为什么用 Welch？** 不要求两组方差相等；大样本有利于均值推断，但重尾下仍需检查区间稳定性。
5. **Bootstrap 抽什么？** 在每个实验组内按客户独立有放回抽取原组同样人数，计算两组均值差，重复后取分位数。
6. **Holm 解决什么？** 同时做两个主要比较会增加至少一次误报的风险，Holm 控制这个家族的错误率。
7. **为什么区间和校正 p 值有时不一致？** 常规区间是单项区间，Holm 是家族校正；报告另给更保守的 Bonferroni 同时区间。
8. **不显著能证明无效吗？** 不能。结合区间看是否仍包含业务上值得关注的正收益或损失；等效结论需要另行设计检验。
9. **一个人群显著、另一个不显著，能定向投放吗？** 不能据此证明组间差异，应看交互检验，再在独立样本验证策略。
10. **怎样确定实验跑多久？** 从业务 MDE、方差、功效和比较数量算样本量，再结合日流量和结果成熟窗口；本数据缺少日流量。
11. **SRM 没问题是否证明实验没问题？** 不是。它检查分组数量，不能排除曝光失败、串组、日志缺失或组间干扰。
12. **收入显著是否就上线？** 还要看实际单位经济性、置信区间、投诉退订等护栏及外推条件。
13. **为什么没做 CUPED？** 缺少可核对的实验前同窗收入。历史年度消费可研究协变量调整，但不能包装成已经验证的 CUPED 方差缩减。
14. **如何迁移到 App 推送？** 可以迁移随机化、指标和决策方法；邮件的效应数值不能直接迁移，需要新的业务实验。
15. **实验中每天看 p 值，显著就停可以吗？** 固定样本检验下不可以；要么遵守预定样本与窗口，要么事先设计有效的序贯方法。

## 你需要自己补充的内容

- 用自己的话录一遍介绍，不照读术语；解释你做过的一个口径检查。
- 回答：如果市场同学只给一张转化率上升的图，你还会要求哪三类证据？
- 选择一个实际应聘岗位，把“后续验证”改写成该岗位的业务语言，保留数据和实验边界。
"""
    (root / "docs/interview.md").write_text(text, encoding="utf-8")
