# 分析结果表索引

这些表由运行入口自动生成；数值可用于独立复核和面试演示。金额单位为美元，结果窗口为两周。CSV 使用 UTF-8 BOM，兼容常见表格软件。

| 文件（位于 reports/tables） | 粒度与内容 |
| --- | --- |
| group_metrics.csv | 每个实验组一行；人数、访问者、购买者、总收入、各项均值和收入样本方差 |
| segment_metrics.csv | 分群维度×子群×实验组；三套独立分群不能跨维度求和 |
| quality.json | 缺失、矛盾、零消费、相同属性记录、拼写规范化与排除记录 |
| randomization.json | 预期比例、三组人数、SRM 卡方统计量与诊断阈值 |
| baseline_balance.csv | 实验组×实验前属性；组内均值、相对对照组 SMD |
| effects.csv | 实验组×指标；总体效应、不确定性与检验结果 |
| interaction_tests.csv | 分群维度；联合 HC3 Wald 交互检验及探索性 Holm 校正 |
| subgroup_effects.csv | 分群维度×子群×邮件；收入差、点态区间、组内人数和购买人数 |
| profit_scenarios.csv | 邮件×假设毛利率×假设触达成本；人均净增量和盈亏平衡成本 |
| sample_size_plan.csv | 假设 MDE×标准差倍数；下一轮每组及三组合计样本量 |

## effects.csv 的关键列

| 列名 | 解释 |
| --- | --- |
| metric | spend为收入、conversion为购买率、visit为访问率 |
| n_treatment / n_control | 两组全部客户人数 |
| mean_treatment / mean_control | 组内人均收入或比例 |
| effect | 实验均值减对照均值；比例结果乘100才是百分点 |
| relative_lift | 实验均值/对照均值−1；零基准时留空 |
| ci_low / ci_high | 未做多重调整的单项区间 |
| p_value | 未做多重调整的双侧p值 |
| p_holm | 仅对两个收入检验做 Holm 校正；辅助指标留空 |
| bootstrap_low / bootstrap_high | 收入用户层级百分位 Bootstrap 的单项区间 |
| family_ci_low / family_ci_high | 收入 Bonferroni 同时区间：默认各97.5%，两项家族至少95%近似覆盖 |
| method / status | 使用的方法和推断是否可用 |

`profit_scenarios.csv` 的 `scenario_signal` 仅表示该假设组合下区间相对零的位置，不是自动上线指令。成本与毛利率不确定性没有包含在区间中。

`sample_size_plan.csv` 的功效针对单个预设比较，使用正态近似且无失访膨胀；应在实际实验前更新近期方差、完整观察比例与流量假设。


## 本阶段交互、子群与成本字段（2026-09-27核对）

| interaction_tests.csv 列 | 含义 |
| --- | --- |
| dimension | 客户类型、历史消费或购买间隔；每行一个维度的联合检验 |
| wald_chi2 / df | HC3联合Wald卡方统计量/限制条件数（2、4、4）；这里df不是客户DataFrame |
| p_value | H0为两种邮件在该维度的全部交互差异同时为0的原始p值 |
| covariance | HC3；允许条件异方差，不能解决客户相关性或选择偏差 |
| p_holm_exploratory | 三个维度联合检验构成的探索性Holm家族，与effects.csv的两个收入检验不同 |

子群表每行是“维度×类别×邮件”相对同类别不发邮件的收入增量。`n_treatment/n_control`包含零消费客户；`buyers_*`是购买人数（质量校验确认spend>0与conversion=1一致）。`ci_low/high`与`p_value`为未校正的子群Welch结果，不是交互差异检验；其中`df`为Welch自由度，和交互表的限制数量不同。

| profit_scenarios.csv 列 | 含义 |
| --- | --- |
| arm | 邮件策略 |
| assumed_margin | 假设毛利率，作用于新增销售收入，未在其中重复扣除触达成本 |
| assumed_contact_cost_usd | 相对不发邮件，每名随机分组客户的增量触达成本 |
| net_per_customer | 收入增量点估计×毛利率−成本 |
| net_ci_low_family / net_ci_high_family | 对收入Bonferroni区间做同样变换；只包含收入抽样不确定性 |
| break_even_cost_point | 收入增量点估计×毛利率，净收益点估计为0时的成本 |
| break_even_cost_lower_bound | 收入区间下界×毛利率，净收益下界为0时的成本，不保证实际盈利 |
| scenario_signal | 区间全正/全负/跨零或触零/不可估计；不是自动推广指令 |

默认30个成本情景为2种邮件×3个毛利率×5个成本。固定假设下区间全正支持条件增量收益，不能核实假设、当前业务效果、其他费用或长期影响。学习中成本.08是手算情景，不在配置默认列表。

样本量表对应总体收入比较；若下一轮检验新老客户增量之差，需另行针对交互MDE、各单元方差及分配比例规划。
