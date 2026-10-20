"""
Task 2: Customer Churn Predictor
Builds a sample customer/order dataset, defines churn signals, scores every customer,
and outputs a ranked at-risk list. Also trains a small logistic regression as a cross-check.
Run:  python churn_predictor.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(7)
TODAY = pd.Timestamp("2026-10-15")
N = 300

# ---------- 1. Sample customers ----------
# Three hidden behaviour groups: loyal, fading (churning soon), already-quiet
group = rng.choice(["loyal", "fading", "quiet"], N, p=[0.65, 0.2, 0.15])
rows = []
for i in range(N):
    g = group[i]
    prior_orders = rng.poisson(6 if g != "quiet" else 4)          # orders in days 31-90 ago
    if g == "loyal":
        recent_orders = max(0, int(prior_orders / 2 + rng.normal(0, 1)))   # last 30 days (per-30-day scale)
        days_inactive = int(rng.integers(1, 20))
    elif g == "fading":
        recent_orders = int(rng.integers(0, 2))
        days_inactive = int(rng.integers(20, 50))
    else:
        recent_orders = 0
        days_inactive = int(rng.integers(45, 120))
    avg_order = round(float(np.clip(rng.normal(60, 25), 10, 200)), 2)
    tickets = int(rng.poisson(0.3 if g == "loyal" else 1.5))
    rows.append((f"C{i+1:04d}", prior_orders, recent_orders, days_inactive, avg_order, tickets,
                 int(g != "loyal")))
df = pd.DataFrame(rows, columns=["customer_id", "orders_prev_60d", "orders_last_30d",
                                 "days_since_last_order", "avg_order_value", "support_tickets", "churned_label"])
# prior window is 60 days -> convert to a per-30-day rate for a fair comparison
df["prev_rate_30d"] = df["orders_prev_60d"] / 2
df.to_csv("data/customers.csv", index=False)

# ---------- 2. Churn signals (rule-based, explainable) ----------
# Signal A: 30+ days inactive
df["sig_inactive"] = (df["days_since_last_order"] >= 30).astype(int)
# Signal B: order frequency dropped 50%+ vs the previous period
df["freq_drop_pct"] = np.where(df["prev_rate_30d"] > 0,
                               (1 - df["orders_last_30d"] / df["prev_rate_30d"]) * 100, 0).round(1)
df["sig_freq_drop"] = (df["freq_drop_pct"] >= 50).astype(int)
# Signal C: 3+ support tickets (extra signal)
df["sig_tickets"] = (df["support_tickets"] >= 3).astype(int)

# Weighted risk score (0-100): inactivity matters most
df["risk_score"] = (50 * df["sig_inactive"] + 35 * df["sig_freq_drop"] + 15 * df["sig_tickets"]
                    + np.clip(df["days_since_last_order"] - 30, 0, 60) / 60 * 10).clip(upper=100).round(1)
df["risk_level"] = pd.cut(df["risk_score"], [-1, 49.9, 74.9, 101], labels=["Low", "Medium", "High"])

def reasons(r):
    out = []
    if r.sig_inactive: out.append(f"{r.days_since_last_order} days inactive")
    if r.sig_freq_drop: out.append(f"order frequency down {r.freq_drop_pct:.0f}%")
    if r.sig_tickets: out.append(f"{r.support_tickets} support tickets")
    return "; ".join(out)
df["why_flagged"] = df.apply(reasons, axis=1)

ranked = (df[df["risk_score"] >= 50]
          .sort_values(["risk_score", "avg_order_value"], ascending=False)
          [["customer_id", "risk_score", "risk_level", "days_since_last_order",
            "orders_last_30d", "freq_drop_pct", "avg_order_value", "why_flagged"]]
          .reset_index(drop=True))
ranked.insert(0, "rank", ranked.index + 1)
ranked.to_csv("outputs/at_risk_customers_ranked.csv", index=False)

# ---------- 3. Cross-check with a simple ML model ----------
feats = ["days_since_last_order", "orders_last_30d", "prev_rate_30d", "support_tickets", "avg_order_value"]
Xtr, Xte, ytr, yte = train_test_split(df[feats], df["churned_label"], test_size=0.25,
                                      random_state=1, stratify=df["churned_label"])
model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
auc_model = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
auc_rules = roc_auc_score(df["churned_label"], df["risk_score"])
print(f"Customers: {N} | flagged at-risk: {len(ranked)} | High risk: {(ranked.risk_level=='High').sum()}")
print(f"AUC rule score: {auc_rules:.2f} | AUC logistic regression (test): {auc_model:.2f}")
open("outputs/model_metrics.txt", "w").write(f"rule_auc={auc_rules:.3f}\nlogreg_auc={auc_model:.3f}\n"
                                    f"customers={N}\nflagged={len(ranked)}\n")

# ---------- 4. Chart ----------
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"])
ax.bar(counts.index.astype(str), counts.values, color=["#4caf7d", "#e8a33d", "#d9534f"])
ax.set_title("Customers by churn risk level", fontweight="bold"); ax.set_ylabel("Customers")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("outputs/risk_distribution.png", dpi=150)
print(ranked.head(5).to_string(index=False))
