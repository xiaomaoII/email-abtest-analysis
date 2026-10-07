# 营销触达的增量收益评估与人群运营建议

面向数据分析求职的公开数据项目：用 SQL 和 Python 复盘三组随机实验，回答“营销带来多少额外收入、证据是否可靠、下一步如何决策”。这是 2008 年历史实验的独立分析，不是本人实施的公司项目。

## 从这里开始

当前状态（2026-10-07）：原理学习与本人业务初稿已有记录；
既有分析实现由 Agent 编写，用户亲手写过部分 SQL。
用户已在逐步指导下编写 [Python 练习笔记本](notebooks/python_practice.ipynb)，P1、P2 已完成指导验收；P2 汇总与 SQL 核对通过，新增未访问人数代码由 Agent 按用户要求补写，用户运行并解释。P2 已提交并推送（`7e6772c`）。P3 已完成两项 Welch、单项区间及 Holm 的指导计算与顺序运行核对；女装比较和汇总为参考后复现，相对提升口径尚待核对。继续如实记录辅助程度。
已在指导下完成 G1 本地版本管理和 G2 首次私有仓库上传，并通过保存、提交、推送的基础概念问答。
参考实现与学习记录已上传到私有仓库 `xiaomaoII/email-abtest-analysis`。
后续按 [Python 独立实操＋Git/GitHub 项目发布计划](docs/python_independent_practice_plan.md) 记录 P3 本轮成果，补充相对提升口径后推进 P4—P6；
G3 将随 P1—P6 持续练习提交和推送，之后完成 G4 分支、Pull Request 与纠错练习，最后进行 G5 新克隆复现和 G6 公开发布。
终端为主、PyCharm 辅助；每次由用户亲手完成一个小任务。
接手先读 [PROJECT_STATE.md](PROJECT_STATE.md) 和 [TODO.md](TODO.md)；
[技术审计](reports/checkpoints/2026-09-27/review.md)验证参考实现，不代表个人代码能力。

1. [一页业务决策摘要](reports/decision_summary.md)：参考产物；本人初稿在 `exercises/my_answers.md`。
2. [完整分析报告](reports/analysis_report.md)：查看口径、证据和适用边界。
3. [分阶段学习路线](docs/learning_path.md)：核对学习步骤与Notebook章节，再查看参考解答。
4. [无输出的展示笔记本](notebooks/analysis_portfolio.ipynb)：保留分析代码，适合在仓库中查看；本地学习版 `notebooks/analysis.ipynb` 已通过 `.gitignore` 排除，不会被删除。
5. [简历与面试材料](docs/interview.md)：完成理解和复现后，改写成自己的表达。

## 环境与运行

建议 Python 3.12；项目在 Windows、Python 3.12 环境验证。当前机器已准备环境和数据。本阶段检查可在项目根目录执行，结果写到终端，不覆盖用户Notebook或正式报告：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe reports/checkpoints/2026-09-27/verify.py
```

以下为首次安装和完整重现的参考流程，需要联网；当前Checkpoint无需执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py all
```

Linux/macOS 将解释器路径改为 `.venv/bin/python`。中文图表需要系统安装微软雅黑、黑体或 Noto Sans CJK SC 等字体。

`requirements.txt` 固定主要依赖；`requirements-lock.txt` 是历史完整环境快照。当前NumPy为2.4.6，文件锁定2.3.5；41项测试与阶段复算通过，但尚未在独立环境完成严格锁版本复现。当前环境未被降级。网络慢时可选择自己可信的 PyPI 镜像；不需要关闭证书验证。

| 命令 | 作用 |
| --- | --- |
| `python run.py fetch` | 获取公开数据，核验行数与字段，记录来源和 SHA256 |
| `python run.py prepare` | 数据质量检查、派生分群、SQL 指标与独立核对 |
| `python run.py analyze` | 重新生成所有结果表、图表、报告和面试材料 |
| `python run.py notebook` | 重建并执行参考笔记本，核对分析结果，需先运行 analyze |
| `python run.py test` | 执行口径、统计方法和异常处理测试 |
| `python run.py all` | 依次完成分析、笔记本执行和测试 |

以上表格中的 `python` 均指项目独立环境的解释器。`run.py` 无参数默认执行 `all`。`all` / `notebook` 会重建并执行整本笔记本，可能覆盖手工修改；当前暂停推进时不要将它们当成下一步任务。模板已同步双表、Bootstrap小数组、区间对照、收入Holm表，以及本阶段六组均值/增量差、用户说明、女装40%成本筛选单元（共29单元），读取 `exercises/01_metrics.sql` 并独立核对；未来新增单元仍需备份并同步。正式流水线使用 `sql/01_metrics.sql`，练习查询通过测试与之对账。自己的笔记另存于 `exercises/my_answers.md`。本次保留现用Notebook及所有保存输出。

学习已从第8节样本量继续到第6节分群、第7节成本和本人业务初稿。步骤13的本人定稿、简历审核与面试演练尚未完成，延后到Python与Git/GitHub实操及发布验收之后。09-24复核脚本保留作历史记录，其25单元和旧哈希断言不适用于当前Notebook；使用上方09-27入口。

## 分析口径

- 原始数据一行代表一名客户，保留三组和全部零消费记录。无用户 ID，不能凭相同属性去重。
- 主指标：两周消费总额 / 全部分组客户数，单位为美元。辅助指标：购买率、访问率，分母相同。
- 两种邮件分别与不发邮件比较。收入用 Welch 检验和用户层级 Bootstrap；对两个主要比较做 Holm 校正。
- 点估计、单项置信区间和多重校正各自标注；收益情景使用保守的 Bonferroni 家族区间。
- 分群只使用实验前属性，并做 HC3 稳健交互检验。该分析是探索，不构成已经验证的定向策略。
- 毛利率、成本与未来 MDE 是显式假设；不能把情景利润当成实际业绩。

## 数据与文件组织

| 位置 | 内容 |
| --- | --- |
| `data/raw/` | 完整原始 CSV 与来源、校验值；CSV 不默认纳入版本控制 |
| `data/processed/` | 派生分析数据和 DuckDB 数据库，可重新生成 |
| `sql/` | 总体指标与分群指标查询 |
| `abtest/` | 数据、统计方法、报告与笔记本的共享实现 |
| `reports/tables/` | 可追溯数值表及审计 JSON |
| `reports/figures/` | 收入效果、实验前平衡、探索性分群图 |
| `docs/` | 业务说明、数据字典、学习路线、参考答案和面试材料 |
| `tests/` | 关键口径、统计方法与失败场景的自动验证 |

所有运行参数集中在 `config.json`。确需完整重新分析且已保护手写修改时，才运行 `all`。最新Checkpoint见 [2026-09-27审计](reports/checkpoints/2026-09-27/audit.json)和[41项测试记录](reports/checkpoints/2026-09-27/tests.xml)。09-24日期目录保留阶段二历史证据。根下 `reports/checkpoint.json` / `checkpoint_tests.xml` 保留2026-09-20记录，`reports/run_manifest.json` 保留2026-09-16完整运行记录；本次没有改写旧日期或刷新正式表图。不同平台浮点数、字体和图像字节可能有细微差异。

## 来源与复现说明

- [Kevin Hillstrom 原始实验说明](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)：数据背景、字段定义与随机分组。
- 原站不可用时，获取 [scikit-uplift 官方代码指定的 Hillstrom 镜像](https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html#fetch_hillstrom)，核验其公开 MD5，再记录解压 CSV 的 SHA256。不会伪装为原站下载，也不关闭 HTTPS 校验。
- 原始数据用途遵守其来源条款；公开说明未给出明确标准许可证，因此提供获取脚本，不默认再分发原始 CSV。
- [SciPy Welch 检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)、[statsmodels 多重检验](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)、[两独立比例差区间](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html)为方法实现参考。

没有逐日数据、送达日志、成本、毛利率和长期护栏；报告不推断当前业务上线收益或长期留存。更详细的分析约定见 [业务说明](docs/business_brief.md)。
