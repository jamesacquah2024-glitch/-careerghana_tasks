# Customer Churn Predictor: Methodology Note

**Goal:** Flag customers likely to leave so the team can reach out before they do.
**Data:** Sample dataset of 300 customers (`customers.csv`) with order counts, days since last order, average order value and support tickets. The data is synthetic, so the results show the method works, not real business performance.

## Churn signals
| Signal | Rule | Weight |
|---|---|---|
| Inactivity | 30+ days since last order | 50 pts |
| Frequency drop | Orders in last 30 days are 50%+ below the previous 30-day rate | 35 pts |
| Support friction | 3+ support tickets | 15 pts |
| Inactivity bonus | Up to +10 pts for each extra day past 30 (capped at 60 days) | 0-10 pts |

Inactivity carries the most weight because a long gap is the clearest sign a customer has stopped buying. The score is capped at 100.

## Risk levels and ranking
- **High:** score 75+. **Medium:** 50-74. **Low:** under 50 (not listed).
- Customers scoring 50+ are ranked by score, with ties broken by average order value so the team contacts the most valuable customers first.
- Each row includes a plain-English `why_flagged` reason, e.g. "42 days inactive; order frequency down 71%".

## Validation
- Rule-based score vs. the sample's churn label: **AUC 0.97**.
- A logistic regression (5 features, 75/25 split) reached **AUC 1.00** on the test set. That is near-perfect because the sample data was built with clearly separated behaviours. Real customer data is messier, so expect lower numbers.
- Results: 93 of 300 customers flagged (80 High, 13 Medium).

## Limitations and next steps
1. Thresholds (30 days, 50%) are starting points and should be tuned per business, since some products are naturally bought every 45 days.
2. Validate against real churn outcomes before relying on the scores.
3. Add signals such as discount usage, email engagement and refund history.

**Recommended action:** Contact High-risk customers within a week with a personalised offer, and monitor Medium-risk customers.
