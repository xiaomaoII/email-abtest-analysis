-- UNION ALL 保留三套独立的分群维度，不应将所有行相加作为总客户数。
WITH segments AS (
    SELECT segment, 'customer_type' AS dimension, customer_type AS subgroup, spend, conversion FROM customers
    UNION ALL
    SELECT segment, 'history_band', history_band, spend, conversion FROM customers
    UNION ALL
    SELECT segment, 'recency_band', recency_band, spend, conversion FROM customers
)
SELECT dimension, subgroup, segment, COUNT(*) AS n,
       AVG(spend) AS revenue_per_customer,
       AVG(conversion) AS conversion_rate
FROM segments
GROUP BY dimension, subgroup, segment
ORDER BY dimension, subgroup, segment;
