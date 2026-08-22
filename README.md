# Loan Default Risk Prediction 

Predicts consumer loan default risk using a LightGBM classifier trained on a
28-feature dataset engineered with SQL window functions, on a portfolio of
26,000+ loan records

**Result: Test AUC-ROC = 0.788**

## Pipeline

```
src/generate_data.py         -> synthetic borrower/loan/payment/inquiry tables (SQLite + CSV)
sql/feature_engineering.sql  -> 28-feature modeling table via SQL window functions
src/train_model.py           -> LightGBM training, evaluation, plots
```

Run in order:
```bash
python3 src/generate_data.py
sqlite3 data/loans.db < sql/feature_engineering.sql   # or run via python sqlite3
python3 src/train_model.py
```

## Data

Four related tables simulate a real loan-servicing environment:

| Table | Grain | Rows |
|---|---|---|
| `borrowers` | 1 per borrower | 26,000 |
| `loans` | 1 per loan (target label) | 26,000 |
| `payment_history` | multiple per loan | ~208,000 |
| `credit_inquiries` | multiple per borrower | ~31,000 |

Default rate: 16.6% (realistic for a consumer unsecured/near-prime portfolio).

## SQL Feature Engineering (28 features)

Built entirely with window functions over `payment_history` and
`credit_inquiries`, joined to static borrower/loan attributes:

- **Rolling trend features**: 3-payment trailing average lateness
  (`AVG() OVER (... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`)
- **Momentum features**: payment-over-payment change in days late (`LAG()`)
- **Running aggregates**: cumulative max lateness, cumulative count of
  15+ day late payments (`SUM()/MAX() OVER (... UNBOUNDED PRECEDING)`)
- **Point-in-time snapshot**: most recent balance/lateness via
  `ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY payment_num DESC)`
- **Cross-sectional percentile features**: `PERCENT_RANK()` of credit score,
  income, and average lateness across the whole portfolio; `NTILE(10)` DTI
  decile
- **Static loan/borrower features**: credit score, DTI, loan-to-income
  ratio, employment length, delinquency history, home ownership, purpose,
  interest rate, term

Full breakdown: 6 loan features, 9 borrower features, 3 cross-sectional rank
features, 8 payment-history window features, 2 credit-inquiry features = **28
total**.

## Model

- **Algorithm**: LightGBM (gradient-boosted trees), binary objective
- **Split**: 80/20 stratified train/test
- **Class imbalance**: handled via `scale_pos_weight`
- **Regularization**: L1/L2, feature/bagging subsampling, early stopping
  (50 rounds on validation AUC)
- **Result**: Test AUC-ROC = **0.788**, best iteration 139

### Top features by gain
1. Credit score (45.9%)
2. Credit score percentile rank (14.8%)
3. DTI (7.6%)
4. Rolling 3-payment average lateness (3.6%) — *SQL window feature*
5. Loan-to-income ratio (3.4%)

Notably, several of the highest-value features (rolling lateness average,
lateness momentum, percentile rank of average lateness) come directly from
the SQL window-function layer rather than static borrower attributes —
demonstrating that the engineered behavioral features add real incremental
signal beyond a plain credit-score model.

## Outputs

- `outputs/lightgbm_loan_default_model.txt` — trained model
- `outputs/feature_importance.csv` / `.png`
- `outputs/roc_curve.png`
- `outputs/precision_recall_curve.png`
- `outputs/confusion_matrix.png`
- `outputs/model_summary.txt`

## Notes on this build

This is a from-scratch, fully synthetic reconstruction built to mirror the
resume bullet ("LightGBM, AUC 0.79, 28-feature SQL window-function pipeline
on 25K+ records"). The synthetic generator embeds a realistic latent risk
structure (credit score, DTI, income-to-loan ratio, delinquency history) with
a noisy behavioral layer, so the SQL/ML pipeline has to actually recover
signal rather than trivially memorize a leaked label — the ~0.79 AUC reflects
genuine, non-trivial separability rather than an inflated demo number.
