# Bayesian Suite (ESS/sec vs Stan + PyMC)

This directory is reserved for the **Bayesian benchmark suite** in the standalone public benchmarks repo.

Canonical methodology + runbook lives in the main docs site:

- `/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/suites/bayesian.md`

Planned measurements (publishable snapshots):

- ESS/sec (bulk + tail) with declared inference settings
- divergence / treedepth saturation rates
- max R-hat, min ESS, min E-BFMI
- wall-time distributions and environment manifest

Seed baselines (today):

- `histfactory_simple_8p` — simple HistFactory workspace (few params)
- `glm_logistic_regression` — synthetic logistic regression
- `hier_random_intercept_non_centered` — synthetic hierarchical logistic random intercepts (non-centered)

Note: the seed reports an ESS/sec proxy computed as `min(ESS_bulk) / wall_time`, where wall time includes warmup + sampling.

Status: runnable seed.

- `nextstat` backend: always available (dependency-light).
- `cmdstanpy` and `pymc` backends: optional (best-effort). If deps are missing, artifacts are emitted as `warn` with an actionable `reason`.

Run (NextStat-only):

```bash
python3 suites/bayesian/suite.py --deterministic --out-dir out/bayesian
```

Run with optional backends:

```bash
python3 suites/bayesian/suite.py --deterministic --out-dir out/bayesian --backends nextstat,cmdstanpy,pymc
```
