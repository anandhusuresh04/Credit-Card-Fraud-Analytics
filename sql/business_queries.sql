-- SQL-only retrospective business analysis (not ML policy validation).

-- name: amount_segment_ranking
WITH bands AS (
 SELECT amount_band, COUNT(*) AS reviewed, SUM(class) AS fraud_cases,
        SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) AS fraud_cents,
        SUM(CASE WHEN class=0 THEN amount_cents ELSE 0 END) AS legitimate_cents
 FROM transactions GROUP BY amount_band
)
SELECT amount_band, reviewed, fraud_cases, reviewed-fraud_cases AS legitimate_reviews,
       ROUND(100.0*fraud_cases/NULLIF(reviewed,0),5) AS observed_review_yield_pct,
       fraud_cents/100.0 AS recorded_fraud_value,
       legitimate_cents/100.0 AS legitimate_value_reviewed,
       DENSE_RANK() OVER (ORDER BY fraud_cents DESC) AS value_exposure_rank,
       DENSE_RANK() OVER (ORDER BY fraud_cases DESC) AS fraud_case_rank
FROM bands ORDER BY amount_band;

-- name: retrospective_band_review_policies
WITH marked AS (
 SELECT t.class, t.amount_cents, p.policy_name,
        CASE WHEN p.policy_name='all' THEN 1
             WHEN p.policy_name='zero_amount' AND t.amount_cents=0 THEN 1
             WHEN p.policy_name='lowest_amount_band' AND t.amount_band=0 THEN 1
             WHEN p.policy_name='highest_amount_band' AND t.amount_band=(SELECT MAX(band_id) FROM dim_amount_band) THEN 1
             WHEN p.policy_name='lowest_or_highest_band' AND t.amount_band IN (0,(SELECT MAX(band_id) FROM dim_amount_band)) THEN 1
             ELSE 0 END AS review
 FROM transactions t CROSS JOIN (
    SELECT 'all' AS policy_name UNION ALL SELECT 'zero_amount'
    UNION ALL SELECT 'lowest_amount_band' UNION ALL SELECT 'highest_amount_band'
    UNION ALL SELECT 'lowest_or_highest_band'
 ) p
), totals AS (
 SELECT SUM(class) AS total_cases,
        SUM(CASE WHEN class=1 THEN amount_cents ELSE 0 END) AS total_fraud_cents
 FROM transactions
)
SELECT m.policy_name, SUM(review) AS reviews,
       SUM(CASE WHEN review=1 AND class=1 THEN 1 ELSE 0 END) AS fraud_cases_reviewed,
       SUM(CASE WHEN review=1 AND class=0 THEN 1 ELSE 0 END) AS legitimate_reviews,
       ROUND(100.0*SUM(CASE WHEN review=1 AND class=1 THEN 1 ELSE 0 END)/NULLIF(SUM(review),0),4) AS observed_review_yield_pct,
       ROUND(100.0*SUM(CASE WHEN review=1 AND class=1 THEN 1 ELSE 0 END)/NULLIF(totals.total_cases,0),4) AS case_coverage_pct,
       ROUND(100.0*SUM(CASE WHEN review=1 AND class=1 THEN amount_cents ELSE 0 END)/NULLIF(totals.total_fraud_cents,0),4) AS value_coverage_pct,
       SUM(CASE WHEN review=1 AND class=0 THEN amount_cents ELSE 0 END)/100.0 AS legitimate_value_reviewed
FROM marked m CROSS JOIN totals
GROUP BY m.policy_name ORDER BY m.policy_name;

-- name: high_value_fraud_concentration
WITH ranked AS (
 SELECT original_row, amount_cents, class,
        ROW_NUMBER() OVER (PARTITION BY class ORDER BY amount_cents DESC, original_row ASC) AS value_rank
 FROM transactions
)
SELECT CASE WHEN value_rank <= 10 THEN 'top_10_fraud_cases' ELSE 'other_fraud_cases' END AS case_group,
       COUNT(*) AS cases,
       SUM(amount_cents)/100.0 AS recorded_fraud_value
FROM ranked WHERE class=1
GROUP BY CASE WHEN value_rank <= 10 THEN 'top_10_fraud_cases' ELSE 'other_fraud_cases' END
ORDER BY case_group DESC;
