"""
train_model.py
----------------
Loads the 28-feature SQL-engineered table, trains a LightGBM binary
classifier to predict loan default, and reports AUC + supporting plots.
"""

import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH = "/home/claude/loan_default_project/data/loans.db"
OUT_DIR = "/home/claude/loan_default_project/outputs"

def load_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()
    return df


def main():
    df = load_features()
    print(f"Loaded features table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    y = df["target"]
    X = df.drop(columns=["loan_id", "target"])
    feature_names = X.columns.tolist()
    print(f"Feature count: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 2.0,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=800,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)],
    )

    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Test AUC-ROC: {auc:.4f}")

    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, y_pred_label, digits=3))

    cm = confusion_matrix(y_test, y_pred_label)

    # ---- Save model ----
    model.save_model(f"{OUT_DIR}/lightgbm_loan_default_model.txt")

    # ---- Feature importance plot ----
    importance = pd.DataFrame({
        "feature": feature_names,
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = 100 * importance["gain"] / importance["gain"].sum()
    importance.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

    plt.figure(figsize=(9, 8))
    top = importance.head(15).iloc[::-1]
    plt.barh(top["feature"], top["gain_pct"], color="#2E5AAC")
    plt.xlabel("Relative Importance (% of total gain)")
    plt.title("Top 15 Feature Importances - LightGBM Loan Default Model")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6.5, 6))
    plt.plot(fpr, tpr, color="#2E5AAC", lw=2, label=f"LightGBM (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Loan Default Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # ---- Confusion matrix plot ----
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (threshold = 0.5)")
    plt.colorbar()
    ticks = ["No Default", "Default"]
    plt.xticks([0, 1], ticks)
    plt.yticks([0, 1], ticks)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Precision-Recall curve ----
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6.5, 6))
    plt.plot(rec, prec, color="#C0392B", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Loan Default Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/precision_recall_curve.png", dpi=150)
    plt.close()

    # ---- Summary file ----
    with open(f"{OUT_DIR}/model_summary.txt", "w") as f:
        f.write("LOAN DEFAULT RISK PREDICTION - MODEL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Records: {df.shape[0]:,}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Train/Test split: 80/20 (stratified)\n")
        f.write(f"Test set size: {len(y_test):,}\n")
        f.write(f"Overall default rate: {y.mean():.2%}\n\n")
        f.write(f"Best boosting round: {model.best_iteration}\n")
        f.write(f"Test AUC-ROC: {auc:.4f}\n\n")
        f.write("Top 10 features by gain:\n")
        for _, row in importance.head(10).iterrows():
            f.write(f"  {row['feature']:<32} {row['gain_pct']:.1f}%\n")

    print(f"\nAll outputs saved to {OUT_DIR}/")
    return auc


if __name__ == "__main__":
    main()
