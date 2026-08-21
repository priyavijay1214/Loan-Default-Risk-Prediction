-- ============================================================================
-- feature_engineering.sql
-- Builds a 28-feature modeling table for loan default prediction using
-- window functions over payment_history and credit_inquiries.
-- Run against data/loans.db (SQLite). Output table: features.
-- ============================================================================

DROP TABLE IF EXISTS features;

CREATE TABLE features AS

WITH

-- ---------------------------------------------------------------------------
-- 1) Payment-level window features: trend & rolling stats per loan
-- ---------------------------------------------------------------------------
payment_windows AS (
    SELECT
        loan_id,
        payment_num,
        days_late,
        remaining_balance,
        payment_amount,

        -- rolling 3-payment average lateness (trailing window)
        AVG(days_late) OVER (
            PARTITION BY loan_id ORDER BY payment_num
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS roll3_avg_days_late,

        -- lateness on the prior payment (lag)
        LAG(days_late, 1) OVER (PARTITION BY loan_id ORDER BY payment_num) AS prev_days_late,

        -- change in lateness vs. prior payment (momentum)
        days_late - LAG(days_late, 1) OVER (PARTITION BY loan_id ORDER BY payment_num) AS days_late_delta,

        -- running max lateness seen so far
        MAX(days_late) OVER (
            PARTITION BY loan_id ORDER BY payment_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_max_days_late,

        -- running count of payments >= 15 days late
        SUM(CASE WHEN days_late >= 15 THEN 1 ELSE 0 END) OVER (
            PARTITION BY loan_id ORDER BY payment_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_late_count,

        -- balance trend: first balance observed for this loan
        FIRST_VALUE(remaining_balance) OVER (
            PARTITION BY loan_id ORDER BY payment_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS first_balance,

        -- rank of this payment within the loan's own history, most recent = 1
        ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY payment_num DESC) AS recency_rank

    FROM payment_history
),

-- ---------------------------------------------------------------------------
-- 2) Collapse payment_windows to one row per loan (take latest snapshot +
--    aggregate trend signals) -- this is the "point-in-time" feature set
-- ---------------------------------------------------------------------------
payment_agg AS (
    SELECT
        loan_id,
        MAX(CASE WHEN recency_rank = 1 THEN roll3_avg_days_late END)      AS latest_roll3_avg_days_late,
        MAX(CASE WHEN recency_rank = 1 THEN running_max_days_late END)    AS max_days_late_ever,
        MAX(CASE WHEN recency_rank = 1 THEN running_late_count END)       AS total_late_payments,
        MAX(CASE WHEN recency_rank = 1 THEN remaining_balance END)        AS latest_balance,
        MAX(first_balance)                                                AS starting_balance,
        AVG(days_late)                                                    AS avg_days_late_alltime,
        AVG(days_late_delta)                                              AS avg_days_late_momentum,
        COUNT(*)                                                          AS n_payments_observed,
        SUM(payment_amount)                                               AS total_paid,
        AVG(payment_amount)                                               AS avg_payment_amount,

        -- percent rank of this loan's average lateness vs. all other loans
        PERCENT_RANK() OVER (ORDER BY AVG(days_late)) AS pctrank_avg_lateness

    FROM payment_windows
    GROUP BY loan_id
),

-- ---------------------------------------------------------------------------
-- 3) Credit inquiry window features per borrower (credit-seeking behavior)
-- ---------------------------------------------------------------------------
inquiry_windows AS (
    SELECT
        ci.borrower_id,
        COUNT(*) AS n_inquiries_24mo,
        SUM(CASE WHEN julianday(l.issue_date) - julianday(ci.inquiry_date) <= 180 THEN 1 ELSE 0 END) AS n_inquiries_6mo,
        AVG(julianday(l.issue_date) - julianday(ci.inquiry_date)) AS avg_days_since_inquiry
    FROM credit_inquiries ci
    JOIN loans l ON l.borrower_id = ci.borrower_id
    GROUP BY ci.borrower_id
),

-- ---------------------------------------------------------------------------
-- 4) Borrower-level cross-sectional rank features (window functions over
--    the whole borrower population -- percentile position, not raw value)
-- ---------------------------------------------------------------------------
borrower_ranks AS (
    SELECT
        borrower_id,
        credit_score,
        annual_income,
        dti,
        open_accounts,
        delinq_2yrs,
        employment_length_yrs,
        home_ownership,
        age,
        PERCENT_RANK() OVER (ORDER BY credit_score)  AS credit_score_pctrank,
        PERCENT_RANK() OVER (ORDER BY annual_income) AS income_pctrank,
        NTILE(10)     OVER (ORDER BY dti)            AS dti_decile
    FROM borrowers
)

-- ---------------------------------------------------------------------------
-- 5) Final assembly: 28 modeling features + loan_id + target
-- ---------------------------------------------------------------------------
SELECT
    l.loan_id,                                                          -- id (not a feature)
    l."default" AS target,                                                -- label

    -- static loan features (6)
    l.loan_amount,
    l.term_months,
    l.interest_rate,
    l.loan_amount / br.annual_income AS loan_to_income_ratio,
    CASE WHEN l.purpose = 'debt_consolidation' THEN 1 ELSE 0 END AS is_debt_consolidation,
    CASE WHEN l.purpose = 'small_business' THEN 1 ELSE 0 END AS is_small_business,

    -- static borrower features (9)
    br.credit_score,
    br.annual_income,
    br.dti,
    br.open_accounts,
    br.delinq_2yrs,
    br.employment_length_yrs,
    br.age,
    CASE WHEN br.home_ownership = 'RENT' THEN 1 ELSE 0 END AS is_renter,
    CASE WHEN br.home_ownership = 'OWN' THEN 1 ELSE 0 END AS owns_home,

    -- borrower cross-sectional rank features (3)
    br.credit_score_pctrank,
    br.income_pctrank,
    br.dti_decile,

    -- payment-history window features (8)
    pa.latest_roll3_avg_days_late,
    pa.max_days_late_ever,
    pa.total_late_payments,
    pa.latest_balance,
    (pa.starting_balance - pa.latest_balance) / (pa.starting_balance + 1) AS pct_balance_paid_down,
    pa.avg_days_late_momentum,
    pa.n_payments_observed,
    pa.pctrank_avg_lateness,

    -- credit inquiry window features (2)
    COALESCE(iw.n_inquiries_24mo, 0) AS n_inquiries_24mo,
    COALESCE(iw.n_inquiries_6mo, 0) AS n_inquiries_6mo

FROM loans l
JOIN borrower_ranks br ON br.borrower_id = l.borrower_id
LEFT JOIN payment_agg pa ON pa.loan_id = l.loan_id
LEFT JOIN inquiry_windows iw ON iw.borrower_id = l.borrower_id;

-- Sanity checks (run separately, not part of CREATE TABLE):
-- SELECT COUNT(*) FROM features;
-- SELECT COUNT(*) AS n_cols FROM pragma_table_info('features');
