"""
Task 4: Budget Threshold Alert Tool
Reads an expense spreadsheet (data/expenses.csv), compares actual spend per category with a set
budget (data/budgets.csv), and sends an alert automatically when a category goes over.

Alert channels (all optional, set as environment variables):
  Email : SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_TO   (e.g. a Gmail app password)
  Slack : SLACK_WEBHOOK_URL
If nothing is configured the tool runs in DRY-RUN mode and saves the alert to outputs/alert_output.txt.
Run:  python budget_alert.py
"""
import os, smtplib, json, urllib.request
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
WARN_AT = 0.90   # warn when 90% of budget is used

# ---------- 1. Sample data (created once) ----------
if not os.path.exists("data/expenses.csv"):
    rng = np.random.default_rng(11)
    cats = {"Marketing": 4500, "Software": 2500, "Travel": 3000, "Office Supplies": 1200,
            "Training": 1800, "Utilities": 1500}
    # make Marketing and Travel overspend, Software near the limit
    targets = {"Marketing": 5320, "Software": 2290, "Travel": 3410, "Office Supplies": 760,
               "Training": 1100, "Utilities": 1240}
    rows = []
    for cat, total in targets.items():
        n = int(rng.integers(6, 10))
        parts = rng.dirichlet(np.ones(n)) * total
        for p in parts:
            day = int(rng.integers(1, 15))
            rows.append((f"2026-10-{day:02d}", cat, round(float(p), 2), f"{cat} expense"))
    pd.DataFrame(rows, columns=["date", "category", "amount", "description"]).sort_values("date") \
      .to_csv("data/expenses.csv", index=False)
    pd.DataFrame({"category": list(cats), "monthly_budget": list(cats.values())}) \
      .to_csv("data/budgets.csv", index=False)

# ---------- 2. Compare actual vs budget ----------
exp, bud = pd.read_csv("data/expenses.csv"), pd.read_csv("data/budgets.csv")
spend = exp.groupby("category")["amount"].sum().round(2).reset_index(name="actual_spend")
report = bud.merge(spend, on="category", how="left").fillna({"actual_spend": 0})
report["pct_used"] = (report["actual_spend"] / report["monthly_budget"] * 100).round(1)
report["over_by"] = (report["actual_spend"] - report["monthly_budget"]).round(2)
report["status"] = np.where(report["pct_used"] > 100, "OVER BUDGET",
                    np.where(report["pct_used"] >= WARN_AT * 100, "WARNING", "OK"))
report.to_csv("outputs/budget_report.csv", index=False)

# ---------- 3. Chart ----------
colors = report["status"].map({"OVER BUDGET": "#d9534f", "WARNING": "#e8a33d", "OK": "#4caf7d"})
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(report))
ax.bar(x - 0.2, report["monthly_budget"], 0.4, label="Budget", color="#c8ccd2")
ax.bar(x + 0.2, report["actual_spend"], 0.4, label="Actual", color=colors)
ax.set_xticks(x); ax.set_xticklabels(report["category"], rotation=20, ha="right")
ax.set_title("Budget vs actual spend (October 2026)", fontweight="bold"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("outputs/budget_vs_actual.png", dpi=150); plt.close()

# ---------- 4. Build alert ----------
alerts = report[report["status"] != "OK"].sort_values("pct_used", ascending=False)
if alerts.empty:
    print("All categories within budget. No alert sent."); raise SystemExit

lines = ["BUDGET ALERT: categories need attention", ""]
for r in alerts.itertuples():
    if r.status == "OVER BUDGET":
        lines.append(f"[OVER BUDGET] {r.category}: ${r.actual_spend:,.2f} spent of ${r.monthly_budget:,.2f} "
                     f"({r.pct_used}%), over by ${r.over_by:,.2f}")
    else:
        lines.append(f"[WARNING] {r.category}: ${r.actual_spend:,.2f} of ${r.monthly_budget:,.2f} "
                     f"({r.pct_used}% used)")
lines += ["", "Please review these departments' spending before month end."]
message = "\n".join(lines)

# ---------- 5. Send ----------
sent = []
if os.getenv("SMTP_HOST") and os.getenv("ALERT_TO"):
    msg = MIMEText(message); msg["Subject"] = f"Budget alert: {len(alerts)} categories flagged"
    msg["From"], msg["To"] = os.environ["SMTP_USER"], os.environ["ALERT_TO"]
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", 587))) as s:
        s.starttls(); s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"]); s.send_message(msg)
    sent.append("email")
if os.getenv("SLACK_WEBHOOK_URL"):
    req = urllib.request.Request(os.environ["SLACK_WEBHOOK_URL"], json.dumps({"text": message}).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req); sent.append("slack")
if not sent:
    open("outputs/alert_output.txt", "w").write(message + "\n")
    sent.append("DRY-RUN (saved to outputs/alert_output.txt)")
print(message); print("\nDelivered via:", ", ".join(sent))
