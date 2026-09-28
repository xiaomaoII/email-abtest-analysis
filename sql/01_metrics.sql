-- 粒度：每行代表一名随机分组客户。source_row_id 仅追溯源文件行，不是真实用户 ID。
-- 分母必须覆盖组内所有客户，不能先过滤 visit = 1 或 conversion = 1。
SELECT
    segment,
    COUNT(*) AS n,
    SUM(visit) AS visitors,
    SUM(conversion) AS buyers,
    SUM(spend) AS revenue,
    AVG(spend) AS revenue_per_customer,
    AVG(visit) AS visit_rate,
    AVG(conversion) AS conversion_rate,
    VAR_SAMP(spend) AS spend_variance
FROM customers
GROUP BY segment
ORDER BY segment;
