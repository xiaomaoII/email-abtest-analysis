-- 练习：在 customers 表上计算三组指标。
-- 要求：输出 segment、客户数、访问人数、购买人数、总收入、访问率、购买率、人均收入。
-- 先写出每个指标的分母，再完成查询。不要过滤访问者或购买者。
-- Notebook 第3节已读取本文件，并独立显示 customers 前5行。
-- 请在下方写自己的 SELECT。
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
