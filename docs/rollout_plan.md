# Rollout plan: from notebook result to live decision

An offline win on last month's data is a hypothesis, not a result. This is how I would take the model live alongside the existing rules engine, and how I would decide whether it actually works.

## Stakeholders

| Who | What they need from this |
|---|---|
| Fraud operations | Alert volume they can staff; reason codes they can act on |
| Product manager (payments) | No drop in payment conversion for genuine customers |
| Engineering | Latency budget (p99 under 100 ms), a rollback switch, versioned model artefacts |
| Risk / compliance | Explainable decisions (reason codes), a documented test, monitoring |
| Customer support | Warning before any change in block volume |

## Phase 1: Shadow mode (2 weeks)

The model scores every live transaction but **blocks nothing**. The API already returns both decisions (`decision` and `rules_decision`), so this is a logging change, not a new build.

What to check:
- **Agreement:** how often do rules and model disagree, and on which segments (product, amount band, new vs returning card)?
- **Volume:** would the model's alert rate fit the ops team's capacity?
- **Latency:** p50/p99 of `/score` under real traffic.
- **Stability:** PSI of the score distribution and of the top features against the training period (`fraud.monitoring.psi`). Above 0.25 means stop and investigate.

Exit: alert volume agreed with ops, latency within budget, no unexplained drift.

## Phase 2: Champion vs challenger A/B test

- **Randomisation unit:** the customer (card), not the transaction. Randomising transactions would put one customer in both arms, and fraudsters adapt across them.
- **Arms:** champion = current rules engine; challenger = model at the cost-tuned threshold. Ramp traffic 5% → 20% → 50% while the guardrails hold.
- **Primary metric:** fraud £ lost per transaction (missed fraud amount + chargeback fee).
- **Guardrails:** genuine-customer block rate, payment completion rate, support contacts about declined payments, API latency.
- **Sample size:** `fraud.experiment.plan_test` computes it from the rules engine's behaviour on the historical test month. The README shows the result for a 10% relative change (α = 0.05, power = 0.8). Because randomisation is by customer while the metric is per transaction, the real variance is higher; analyse with cluster-robust standard errors (or the delta method on per-customer ratios), and treat the calculated n as a lower bound.
- **Variance reduction:** CUPED, using each customer's pre-experiment spend and fraud-score history as the covariate.
- **Decide the analysis before starting:** fixed duration from the power calculation, or a sequential design with pre-set boundaries. No peeking at daily p-values.

### The catch: delayed labels
Fraud is confirmed through chargebacks, which arrive weeks after the transaction. The primary metric is only readable well after the test ends. So during the test, monitor:
- **Leading indicators:** customer-reported fraud, confirmed account-takeover cases, analyst review outcomes.
- **Guardrails:** these are immediate and can trigger a rollback on their own.

The final read on £ loss happens once the chargeback window for the test period has closed.

## Rollback criteria (agreed before launch)
- Genuine block rate in the challenger arm significantly above champion by more than the agreed tolerance.
- p99 latency above budget for more than 15 minutes.
- Score PSI above 0.25 without a known cause.

Rollback = route 100% of traffic to the rules engine (a config flag, no redeploy).

## After launch
- Weekly PSI on scores and top features; monthly retrain on a rolling window with the same time-based validation.
- Re-run the threshold choice whenever the cost assumptions change (for example, a new chargeback fee or reimbursement rule).
- Keep a small holdout (e.g. 2% on the rules engine) so the model's value can still be measured months later.
