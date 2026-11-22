-- SQLite | Query blocks identified by -- name: ...; executed separately.
-- All amounts in source dataset units. Time = elapsed seconds, NOT clock hours.

-- name: executive_kpis
SELECT COUNT(*) AS transactions,
       SUM(class) AS fraud_cases,
       COUNT(*) - SUM(class) AS legitimate_cases,
       ROUND(100.0 * SUM(class) / COUNT(*), 6) AS fraud_rate_pct,
       SUM(amount_cents) / 100.0 AS total_transaction_value,
       SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) / 100.0 AS fraudulent_value,
       SUM(is_exact_duplicate) AS exact_duplicate_rows,
       SUM(CASE WHEN amount_cents=0 THEN 1 ELSE 0 END) AS zero_amount_rows
FROM transactions;

-- name: zero_amount_segment
WITH segment AS (
    SELECT CASE WHEN amount_cents=0 THEN 'zero' ELSE 'positive' END AS amount_type,
           COUNT(*) AS transactions, SUM(class) AS fraud_cases,
           SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) AS fraud_cents
    FROM transactions
    GROUP BY CASE WHEN amount_cents=0 THEN 'zero' ELSE 'positive' END
)
SELECT amount_type, transactions, fraud_cases,
       ROUND(100.0 * fraud_cases / NULLIF(transactions,0), 6) AS fraud_rate_pct,
       fraud_cents / 100.0 AS fraudulent_value
FROM segment ORDER BY amount_type DESC;

-- name: duplicate_audit
SELECT is_exact_duplicate, COUNT(*) AS transactions,
       SUM(class) AS fraud_cases,
       SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) / 100.0 AS fraudulent_value
FROM transactions GROUP BY is_exact_duplicate ORDER BY is_exact_duplicate;

-- name: amount_band_exposure
WITH by_band AS (
    SELECT d.band_id, d.label, COUNT(t.original_row) AS transactions,
           COALESCE(SUM(t.class),0) AS fraud_cases,
           COALESCE(SUM(t.amount_cents),0) AS total_cents,
           COALESCE(SUM(CASE WHEN t.class=1 THEN t.amount_cents ELSE 0 END),0) AS fraud_cents
    FROM dim_amount_band d LEFT JOIN transactions t ON t.amount_band=d.band_id
    GROUP BY d.band_id,d.label
), enriched AS (
    SELECT *, SUM(fraud_cases) OVER () AS all_fraud_cases,
              SUM(fraud_cents) OVER () AS all_fraud_cents
    FROM by_band
)
SELECT band_id, label, transactions, fraud_cases,
       ROUND(100.0 * fraud_cases / NULLIF(transactions,0),6) AS fraud_rate_pct,
       total_cents / 100.0 AS transaction_value,
       fraud_cents / 100.0 AS fraudulent_value,
       ROUND(100.0 * fraud_cases / NULLIF(all_fraud_cases,0),4) AS share_of_fraud_cases_pct,
       ROUND(100.0 * fraud_cents / NULLIF(all_fraud_cents,0),4) AS share_of_fraud_value_pct
FROM enriched ORDER BY band_id;

-- name: amount_band_cumulative_exposure
WITH summary AS (
    SELECT amount_band AS band_id, COUNT(*) AS transactions,
           SUM(class) AS fraud_cases,
           SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) AS fraud_cents
    FROM transactions GROUP BY amount_band
), running AS (
    SELECT band_id, transactions, fraud_cases, fraud_cents,
           SUM(fraud_cases) OVER (ORDER BY band_id ROWS UNBOUNDED PRECEDING) AS cumulative_fraud_cases,
           SUM(fraud_cents) OVER (ORDER BY band_id ROWS UNBOUNDED PRECEDING) AS cumulative_fraud_cents,
           SUM(fraud_cases) OVER () AS total_fraud_cases,
           SUM(fraud_cents) OVER () AS total_fraud_cents
    FROM summary
)
SELECT band_id, transactions, fraud_cases, fraud_cents/100.0 AS fraudulent_value,
       cumulative_fraud_cases,
       ROUND(100.0*cumulative_fraud_cases/NULLIF(total_fraud_cases,0),4) AS cumulative_case_pct,
       cumulative_fraud_cents/100.0 AS cumulative_fraudulent_value,
       ROUND(100.0*cumulative_fraud_cents/NULLIF(total_fraud_cents,0),4) AS cumulative_value_pct
FROM running ORDER BY band_id;

-- name: elapsed_time_exposure
WITH t AS (
    SELECT d.interval_id, d.label, COUNT(x.original_row) AS transactions,
           COALESCE(SUM(x.class),0) AS fraud_cases,
           COALESCE(SUM(CASE WHEN x.class=1 THEN x.amount_cents ELSE 0 END),0) AS fraud_cents
    FROM dim_time_interval d LEFT JOIN transactions x ON x.time_interval=d.interval_id
    GROUP BY d.interval_id,d.label
)
SELECT interval_id, label, transactions, fraud_cases,
       ROUND(100.0*fraud_cases/NULLIF(transactions,0),6) AS fraud_rate_pct,
       fraud_cents/100.0 AS fraudulent_value,
       LAG(fraud_cases) OVER (ORDER BY interval_id) AS prior_interval_fraud_cases
FROM t ORDER BY interval_id;

-- name: feature_class_profile
SELECT class, COUNT(*) AS transactions,
       ROUND(AVG(V14),6) AS mean_v14,
       ROUND(AVG(V17),6) AS mean_v17,
       ROUND(AVG(V12),6) AS mean_v12,
       ROUND(AVG(V10),6) AS mean_v10,
       ROUND(AVG(V7),6) AS mean_v7
FROM transactions GROUP BY class ORDER BY class;
