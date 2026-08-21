"""
generate_data.py
-----------------
Generates a synthetic but realistically-correlated loan portfolio:
  - borrowers            (1 row per borrower)
  - loans                (1 row per loan, target = default flag)
  - payment_history       (multiple rows per loan -> feeds window-function features)
  - credit_inquiries      (multiple rows per borrower -> feeds window-function features)

The underlying "true risk" model is a weighted combination of credit score,
DTI, income-to-loan ratio, recent delinquency behavior, and credit-seeking
behavior, passed through a logistic function with noise. This gives the
downstream SQL feature pipeline + LightGBM model real signal to recover,
similar to a real bureau/loan-servicing dataset.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
N_LOANS = 26000  # >25K records as in resume bullet

STATES = ["TX", "CA", "NY", "FL", "IL", "OH", "GA", "NC", "PA", "MI", "WA", "AZ", "MA", "CO", "VA"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "medical", "small_business", "car", "other"]
HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN"]
GRADES = ["A", "B", "C", "D", "E", "F", "G"]

def generate_borrowers(n):
    borrower_id = np.arange(1, n + 1)
    age = np.clip(RNG.normal(38, 11, n), 21, 75).astype(int)
    annual_income = np.clip(RNG.lognormal(mean=10.9, sigma=0.45, size=n), 18000, 450000).round(2)
    employment_length = np.clip(RNG.exponential(4.5, n), 0, 40).round(1)
    credit_score = np.clip(RNG.normal(680, 65, n), 300, 850).astype(int)
    home_ownership = RNG.choice(HOME_OWNERSHIP, n, p=[0.45, 0.40, 0.15])
    state = RNG.choice(STATES, n)
    dti = np.clip(RNG.normal(22, 10, n), 0, 60).round(2)  # debt-to-income %
    open_accounts = np.clip(RNG.poisson(8, n), 1, 30)
    delinq_2yrs = RNG.poisson(0.35, n)

    return pd.DataFrame({
        "borrower_id": borrower_id,
        "age": age,
        "annual_income": annual_income,
        "employment_length_yrs": employment_length,
        "credit_score": credit_score,
        "home_ownership": home_ownership,
        "state": state,
        "dti": dti,
        "open_accounts": open_accounts,
        "delinq_2yrs": delinq_2yrs,
    })


def generate_loans(borrowers):
    n = len(borrowers)
    loan_id = np.arange(1, n + 1)

    loan_amount = np.clip(RNG.lognormal(mean=9.3, sigma=0.55, size=n), 1000, 40000).round(0)
    term_months = RNG.choice([36, 60], n, p=[0.7, 0.3])
    purpose = RNG.choice(PURPOSES, n, p=[0.28, 0.22, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08])

    # interest rate driven inversely by credit score, plus noise
    base_rate = 22 - (borrowers["credit_score"].values - 300) / 550 * 16
    interest_rate = np.clip(base_rate + RNG.normal(0, 1.8, n), 5.0, 30.0).round(2)

    grade_idx = np.clip(((interest_rate - 5) / 25 * 7).astype(int), 0, 6)
    grade = np.array(GRADES)[grade_idx]

    start = datetime(2021, 1, 1)
    issue_date = [start + timedelta(days=int(d)) for d in RNG.integers(0, 1460, n)]

    # ---- true underlying risk score (latent) ----
    z = (
        -0.020 * (borrowers["credit_score"].values - 680)
        + 0.045 * (borrowers["dti"].values - 22)
        + 0.60 * np.log1p(loan_amount / borrowers["annual_income"].values * 12)
        + 0.35 * borrowers["delinq_2yrs"].values
        - 0.015 * borrowers["employment_length_yrs"].values
        + 0.10 * (term_months == 60).astype(float)
        + 0.25 * (purpose == "small_business").astype(float)
        + 0.15 * (borrowers["home_ownership"].values == "RENT").astype(float)
        - 0.10 * (borrowers["home_ownership"].values == "OWN").astype(float)
        + RNG.normal(0, 1.1, n)  # idiosyncratic noise caps max achievable AUC ~ realistic range
    )
    prob_default = 1 / (1 + np.exp(-(z - 3.35)))
    default = RNG.binomial(1, prob_default)

    loans = pd.DataFrame({
        "loan_id": loan_id,
        "borrower_id": borrowers["borrower_id"].values,
        "loan_amount": loan_amount,
        "term_months": term_months,
        "interest_rate": interest_rate,
        "grade": grade,
        "purpose": purpose,
        "issue_date": [d.strftime("%Y-%m-%d") for d in issue_date],
        "default": default,
    })
    # underlying continuous risk signal (NOT the realized binary label) -
    # used only to weakly, noisily drive payment behavior so payment-history
    # features correlate with true risk without leaking the final label
    latent_risk = prob_default
    return loans, issue_date, latent_risk


def generate_payment_history(loans, issue_dates, latent_risk):
    """
    Simulate up to 12 monthly payments per loan. Payment behavior (lateness
    drift, balance paydown) is driven *noisily* by each loan's underlying
    continuous risk score (not the realized binary default label), so
    features correlate with true default risk probabilistically rather than
    encoding the outcome directly. This avoids unrealistic near-perfect
    separation while still giving the SQL window-function layer real signal.
    """
    rows = []
    n = len(loans)
    loan_amount = loans["loan_amount"].values
    loan_id = loans["loan_id"].values

    n_payments = RNG.integers(4, 13, n)  # 4-12 observed payments

    for i in range(n):
        months = n_payments[i]
        bal = loan_amount[i]
        monthly_pay = loan_amount[i] / 24  # rough amortization proxy

        # risk-linked drift, but heavily noised so it's probabilistic, not deterministic
        risk_component = (latent_risk[i] - 0.17) * 3.0
        drift = RNG.normal(risk_component, 0.85)

        base_date = issue_dates[i]
        days_late_running = max(0, RNG.normal(1.5, 1.5))
        for m in range(1, months + 1):
            days_late_running = max(0, days_late_running + drift + RNG.normal(0, 2.5))
            days_late = max(0, min(90, days_late_running + RNG.normal(0, 3)))
            actual_pay = monthly_pay * (1 - min(days_late, 60) / 220) * RNG.normal(1, 0.05)
            bal = max(0, bal - actual_pay * 0.85)
            pay_date = base_date + timedelta(days=30 * m)
            rows.append((loan_id[i], m, pay_date.strftime("%Y-%m-%d"),
                         round(float(actual_pay), 2), round(float(days_late), 1), round(float(bal), 2)))

    return pd.DataFrame(rows, columns=[
        "loan_id", "payment_num", "payment_date", "payment_amount", "days_late", "remaining_balance"
    ])


def generate_credit_inquiries(borrowers, issue_dates_by_borrower):
    """Recent hard-inquiry events per borrower in the 24 months before loan issuance."""
    rows = []
    for bid, ref_date in issue_dates_by_borrower.items():
        n_inq = RNG.poisson(1.2)
        for _ in range(n_inq):
            days_before = RNG.integers(1, 730)
            inq_date = ref_date - timedelta(days=int(days_before))
            rows.append((bid, inq_date.strftime("%Y-%m-%d")))
    return pd.DataFrame(rows, columns=["borrower_id", "inquiry_date"])


def main():
    print(f"Generating {N_LOANS} borrowers/loans...")
    borrowers = generate_borrowers(N_LOANS)
    loans, issue_dates, latent_risk = generate_loans(borrowers)
    payments = generate_payment_history(loans, issue_dates, latent_risk)
    ref_dates = dict(zip(borrowers["borrower_id"], issue_dates))
    inquiries = generate_credit_inquiries(borrowers, ref_dates)

    borrowers.to_csv("/home/claude/loan_default_project/data/borrowers.csv", index=False)
    loans.to_csv("/home/claude/loan_default_project/data/loans.csv", index=False)
    payments.to_csv("/home/claude/loan_default_project/data/payment_history.csv", index=False)
    inquiries.to_csv("/home/claude/loan_default_project/data/credit_inquiries.csv", index=False)

    conn = sqlite3.connect("/home/claude/loan_default_project/data/loans.db")
    borrowers.to_sql("borrowers", conn, if_exists="replace", index=False)
    loans.to_sql("loans", conn, if_exists="replace", index=False)
    payments.to_sql("payment_history", conn, if_exists="replace", index=False)
    inquiries.to_sql("credit_inquiries", conn, if_exists="replace", index=False)
    conn.close()

    print(f"loans: {len(loans):,} | payments: {len(payments):,} | inquiries: {len(inquiries):,}")
    print(f"Default rate: {loans['default'].mean():.3%}")
    print("Saved CSVs + SQLite DB to data/")


if __name__ == "__main__":
    main()
