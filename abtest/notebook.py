"""Build an executable, pedagogical notebook from the shared analysis functions."""
from __future__ import annotations

import os
from pathlib import Path


def build_notebook(root: Path, execute=True):
    for key, relative in [("JUPYTER_RUNTIME_DIR", ".cache/jupyter/runtime"),
                          ("JUPYTER_CONFIG_DIR", ".cache/jupyter/config"),
                          ("IPYTHONDIR", ".cache/ipython")]:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    import sys

    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("# 营销触达的增量收益评估\n\n公开数据独立复盘，非本人执行的线上实验。金额为美元、观察窗口为两周。\n\n建议先做 `docs/learning_path.md` 中对应练习，再运行参考代码。本笔记本逐步重新计算关键表，并与运行入口产生的结果核对。"),
        md("## 1. 环境与分析约定\n\n主指标：全部随机分组客户的人均收入。辅助指标：购买率、访问率。双侧比较两种邮件各自与不发邮件的差异；主指标采用 Holm 校正。这是复盘约定，不是原实验事前注册。"),
        code("from pathlib import Path\nimport sys, json\nROOT = Path.cwd()\nif not (ROOT / 'config.json').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'config.json').exists(), '请从项目根目录或 notebooks 目录运行'\nsys.path.insert(0, str(ROOT))\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nfrom abtest.data import download, validate, prepare, sql_metrics, reconcile, ARMS\nfrom abtest.statistics import *\nconfig = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))\nvalidate_config(config)\nprint('Python:', sys.executable)\nconfig"),
        md("## 2. 质量检查\n\n无真实用户 ID，不能按属性相同直接去重。缺失结果不能填零。`source_row_id` 只是原文件行号。"),
        code("raw = pd.read_csv(download(ROOT))\nquality = validate(raw)\ndf = prepare(raw)\ndisplay(pd.DataFrame({'dtype': raw.dtypes.astype(str), 'missing': raw.isna().sum()}))\nquality"),
        md("## 3. SQL 汇总与独立核对\n\n先回答：如果先过滤 `visit=1`，改变了什么？不要将实验后筛选得到的人群效果解释为全体客户效果。"),
        code("# 表 1：查看 customers 原始表的前 5 行。\npreview_sql = 'SELECT * FROM customers LIMIT 5;'\ndisplay(sql_metrics(df, preview_sql))\n\n# 表 2：你在 exercises/01_metrics.sql 中写的三组指标汇总。\nsql = (ROOT / 'exercises/01_metrics.sql').read_text(encoding='utf-8')\nprint(sql)\nmetrics = sql_metrics(df, sql)\nreconcile(df, metrics)\ndisplay(metrics)"),
        md("## 4. 分流比例与实验前平衡\n\nSRM 检验阈值是诊断约定。没有异常不证明分流与数据采集全过程正确。SMD 的 0.1 是经验参考线。"),
        code("srm, balance = randomization_checks(df, config)\ndisplay(srm)\ndisplay(balance.sort_values('abs_smd', ascending=False).head(12))"),
        md("## 5. 收入效果与稳健性\n\nWelch 允许方差不同；Bootstrap 在两组内分别重抽客户，包含零消费。报告绝对增量与区间，再看相对提升。"),
        code("effects = experiment_effects(df, config)\ndisplay(effects[effects.metric == 'spend'])\n# 用 scipy 独立核对 Welch 的 p 值。\nfrom scipy.stats import ttest_ind\nfor arm in ARMS[1:]:\n    reference = ttest_ind(df.loc[df.segment == arm, 'spend'], df.loc[df.segment == ARMS[0], 'spend'], equal_var=False).pvalue\n    actual = effects.loc[(effects.arm == arm) & (effects.metric == 'spend'), 'p_value'].iloc[0]\n    np.testing.assert_allclose(actual, reference, rtol=1e-10)"),
        md("### 多重比较\n\n`p_holm` 用于两项收入检验。`ci_low/ci_high` 是单项区间，`family_ci_*` 为更保守的 Bonferroni 家族区间，不应混淆。辅助指标仅用于解释。"),
        code("display(effects[effects.metric != 'spend'][['arm','metric','mean_control','mean_treatment','effect','ci_low','ci_high','p_value']])\ndisplay(Image(filename=str(ROOT / 'reports/figures/revenue_effects.png')))"),
        md("## 6. 探索性人群分析\n\n先定义三个实验前维度，再以 HC3 联合 Wald 交互检验探索效果差异。交互检验与具体人群有效性是不同问题，不能用当前样本同时筛选和验证策略。"),
        code("interactions, subgroups = heterogeneity(df, config['alpha'])\ndisplay(interactions)\ndisplay(subgroups[['dimension','subgroup','arm','n_treatment','buyers_treatment','n_control','buyers_control','effect','ci_low','ci_high']])"),
        md("## 7. 成本情景\n\n这些毛利率与触达成本是假设，数据不含实际利润。净收益下界基于 Bonferroni 区间，仅包含收入估计的不确定性。"),
        code("scenarios = profit_scenarios(effects, config)\ndisplay(scenarios.head(15))\nscenarios.groupby(['arm','scenario_signal']).size()"),
        md("## 8. 下一轮样本量\n\n采用历史最大组内标准差作为规划输入，另做标准差放大情景。MDE 为业务值得识别的绝对收入差，功效针对单个预设比较。不是事后功效。"),
        code("planning = sample_size_plan(metrics, config)\nplanning"),
        md("## 9. 结果追溯与决策\n\n同一配置和随机种子应复现效果表；如你改了配置，请先重新运行分析入口，再运行笔记本。"),
        code("saved = pd.read_csv(ROOT / 'reports/tables/effects.csv')\nkeys = ['arm','metric']\nleft = effects.set_index(keys).sort_index()\nright = saved.set_index(keys).sort_index()\ncolumns = ['effect','p_value','p_holm','ci_low','ci_high','bootstrap_low','bootstrap_high','family_ci_low','family_ci_high']\nnp.testing.assert_allclose(left[columns], right[columns], rtol=1e-10, atol=1e-12, equal_nan=True)\nprint('重算结果与报告效果表一致。')\ndisplay(Markdown((ROOT / 'reports/decision_summary.md').read_text(encoding='utf-8')))"),
        md("## 10. 自测\n\n1. 用一句话说清这项实验估计的是什么。\n2. 如果转化率上升但收入区间跨零，你会怎么建议？\n3. 为什么不能从两项检验的显著性差异判断两种邮件的优劣？\n4. 如何把探索性人群结论转成下一轮可验证假设？\n5. 哪些信息缺失阻止你声称该方案已盈利？"),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"hillstrom-{i:02d}"
    # Assign the original IDs before inserting the learner's saved exercises.
    # Keep the user's existing IDs and order so rebuilding does not lose them.
    practice_after = {
        "hillstrom-10": [
            code("small_treatment = [0, 0, 0, 10, 20]\nsmall_control = [0, 0, 0, 0, 10]\n\npoint_estimate = np.mean(small_treatment) - np.mean(small_control)\n\nbootstrap_ci = bootstrap_difference(\n    small_treatment,\n    small_control,\n    reps=5000,\n    seed=20260917\n)\n\npoint_estimate, bootstrap_ci", id="c7338eb7ce2c522c"),
            code('effects[effects["metric"] == "spend"][\n    ["arm", "effect", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high"]\n]', id="3ede67f86ce0a3f4"),
        ],
        "hillstrom-12": [
            code('effects[effects["metric"] == "spend"][\n    ["arm", "p_value", "p_holm"]\n].sort_values("p_value")', id="763175ad8d71773c"),
        ],
        "hillstrom-14": [
            code('# 按“客户类型 × 邮件策略”计算平均收入\ngroup_means = (\n    df.groupby(["customer_type", "segment"], observed=True)["spend"]\n      .mean()\n      .unstack("segment")\n)\n\n# 在同一种客户类型内部，与不发邮件组比较\ngroup_means["男装邮件增量"] = (\n    group_means["Mens E-Mail"] - group_means["No E-Mail"]\n)\n\ngroup_means["女装邮件增量"] = (\n    group_means["Womens E-Mail"] - group_means["No E-Mail"]\n)\n\ndisplay(group_means)', id="992ae00247d4e0af"),
            md('判断男（女）装邮件对新客户和已有客户的效果是否不同？（近12月新客户男（女）装邮件增量-已有客户男（女）装邮件增量）', id="7c5d9288a25c4670"),
            code('effect_difference = (\n    group_means.loc["近12月新客户", ["男装邮件增量", "女装邮件增量"]]\n    - group_means.loc["已有客户", ["男装邮件增量", "女装邮件增量"]]\n)\n\ndisplay(effect_difference)', id="98388394faf70936"),
        ],
        "hillstrom-16": [
            code('# 根据收入效果、假设毛利率和触达成本，生成所有情景\nscenarios = profit_scenarios(effects, config)\n\n# 只看“女装邮件、毛利率40%”的情景\nwomen_40 = scenarios.loc[\n    (scenarios["arm"] == "Womens E-Mail")\n    & (scenarios["assumed_margin"] == 0.4),\n    [\n        "assumed_contact_cost_usd",\n        "net_per_customer",\n        "net_ci_low_family",\n        "net_ci_high_family",\n        "scenario_signal",\n    ],\n]\n\ndisplay(women_40)', id="3998930164f06feb"),
        ],
    }
    cells = [item for cell in cells for item in [cell, *practice_after.get(cell.id, [])]]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (project .venv)", "language": "python", "name": "python3"}})
    path = root / "notebooks/analysis.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbf.write(nb, path)
    if execute:
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            client = NotebookClient(nb, km=manager, timeout=240, resources={"metadata": {"path": str(root)}})
            client.execute()
            nbf.write(nb, path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print("分析笔记本已逐格执行，输出已保存。", flush=True)
    return path
