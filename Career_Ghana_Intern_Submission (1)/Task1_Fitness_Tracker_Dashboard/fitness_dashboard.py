"""
Task 1: Fitness Tracker Insights Dashboard
Generates a sample dataset (120 users x 56 days), then builds:
  - weekly trend charts (steps, sleep, calories)
  - a "Most Active Users" top-10 leaderboard
  - a flag for users whose activity has dropped 3+ days in a row
Run:  python fitness_dashboard.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(42)
N_USERS, N_DAYS = 120, 56
dates = pd.date_range("2026-08-17", periods=N_DAYS, freq="D")  # starts on a Monday

# ---------- 1. Build sample dataset ----------
rows = []
decliners = set(rng.choice(N_USERS, 12, replace=False))  # users who fade out at the end
for u in range(N_USERS):
    base_steps = float(np.clip(rng.normal(8000, 2500), 2500, 16000))
    base_sleep = float(np.clip(rng.normal(7.0, 0.8), 4.5, 9))
    base_cal = float(np.clip(rng.normal(2200, 300), 1500, 3200))
    for i, d in enumerate(dates):
        weekend = d.dayofweek >= 5
        steps = base_steps * (1.1 if weekend else 1) + rng.normal(0, 1500)
        sleep = base_sleep + (0.5 if weekend else 0) + rng.normal(0, 0.6)
        cal = base_cal + rng.normal(0, 150) + steps * 0.02
        if u in decliners and i >= N_DAYS - 5:           # drop over the last 5 days
            steps *= 1 - 0.15 * (i - (N_DAYS - 6))
        rows.append((f"U{u+1:03d}", d, max(int(steps), 0), round(float(np.clip(sleep, 3, 11)), 1), int(cal)))
df = pd.DataFrame(rows, columns=["user_id", "date", "steps", "sleep_hours", "calories"])
df.to_csv("data/fitness_data.csv", index=False)

# ---------- 2. Weekly trends ----------
df["week"] = ((df["date"] - dates[0]).dt.days // 7 + 1)
weekly = df.groupby("week")[["steps", "sleep_hours", "calories"]].mean().round(1)
weekly.to_csv("outputs/weekly_trends.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color, title in zip(
    axes, ["steps", "sleep_hours", "calories"],
    ["#1b7f5c", "#4a5bd4", "#d9822b"],
    ["Avg daily steps", "Avg sleep (hours)", "Avg daily calories"],
):
    ax.plot(weekly.index, weekly[col], marker="o", color=color, linewidth=2.2)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Week"); ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("outputs/weekly_trends.png", dpi=150); plt.close()

# ---------- 3. Most Active Users (top 10 by avg daily steps) ----------
board = (df.groupby("user_id")["steps"].mean().round(0).astype(int)
         .sort_values(ascending=False).head(10).reset_index())
board.columns = ["user_id", "avg_daily_steps"]
board.insert(0, "rank", range(1, 11))
board.to_csv("outputs/top10_leaderboard.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(board["user_id"][::-1], board["avg_daily_steps"][::-1], color="#1b7f5c")
ax.set_title("Most Active Users (top 10)", fontweight="bold"); ax.set_xlabel("Avg daily steps")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("outputs/leaderboard.png", dpi=150); plt.close()

# ---------- 4. Flag: activity dropped 3+ days in a row ----------
# "Dropped" = steps lower than the previous day. Flag users whose CURRENT streak is 3+ days.
def current_drop_streak(steps):
    streak = 0
    for i in range(len(steps) - 1, 0, -1):
        if steps[i] < steps[i - 1]:
            streak += 1
        else:
            break
    return streak

flags = []
for uid, g in df.sort_values("date").groupby("user_id"):
    s = g["steps"].to_numpy()
    streak = current_drop_streak(s)
    if streak >= 3:
        flags.append((uid, streak, int(s[-streak - 1]), int(s[-1]),
                      round((s[-1] / s[-streak - 1] - 1) * 100, 1)))
flagged = pd.DataFrame(flags, columns=["user_id", "days_dropping", "steps_before_drop", "steps_latest", "pct_change"])
flagged = flagged.sort_values("pct_change").reset_index(drop=True)
flagged.to_csv("outputs/flagged_users.csv", index=False)

# ---------- 5. Top insight + dashboard data ----------
dow = df.assign(dow=df["date"].dt.day_name()).groupby("dow")["steps"].mean()
best_day, worst_day = dow.idxmax(), dow.idxmin()
insight = (f"Activity peaks on {best_day}s (~{dow.max():,.0f} steps) and is lowest on {worst_day}s "
           f"(~{dow.min():,.0f}). {len(flagged)} of {N_USERS} users have dropped activity 3+ days in a row.")
open("outputs/top_insight.txt", "w").write(insight + "\n")

dash = {
    "weeks": weekly.index.tolist(),
    "steps": weekly["steps"].tolist(), "sleep": weekly["sleep_hours"].tolist(), "calories": weekly["calories"].tolist(),
    "leaderboard": board.to_dict("records"), "flagged": flagged.to_dict("records"),
    "dow": {k: round(v) for k, v in dow.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).items()},
    "n_users": N_USERS, "n_days": N_DAYS, "insight": insight,
}
json.dump(dash, open("outputs/dashboard_data.json", "w"))
print(insight); print(board.head(3)); print(flagged.head())
