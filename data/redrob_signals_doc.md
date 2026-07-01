# Redrob Behavioral Signals

Each candidate may include a `redrob_signals` object with platform-derived behavioral metrics.

| Signal | Type | Range | Higher is better? | Description |
|--------|------|-------|-------------------|-------------|
| `profile_views` | int | 0–10000 | Yes | Recruiter/profile view count |
| `response_rate` | float | 0.0–1.0 | Yes | Fraction of recruiter messages answered |
| `avg_tenure_months` | float | 0–600 | Context-dependent | Average months per role; senior roles prefer ≥24 |
| `job_hop_score` | float | 0.0–1.0 | No | Higher = more frequent job changes |
| `engagement_score` | float | 0.0–1.0 | Yes | Overall platform engagement |
| `profile_completeness` | float | 0.0–1.0 | Yes | Fraction of profile sections filled |

## Scoring notes

- Signals are normalized to [0, 1] during preprocessing (winsorized min-max).
- Missing signals impute to neutral 0.5 at ranking time.
- Senior JD personas penalize high `job_hop_score`; client-facing roles boost `response_rate`.

## Honeypot indicators

- Perfect signal values (all ≥ 0.99) combined with low `profile_completeness` (< 0.3) suggests gaming.
- Anomalous `profile_views` with empty career history is suspicious.
