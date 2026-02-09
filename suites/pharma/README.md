# Pharma Suite (Seed)

This is a minimal **pharmacometrics** benchmark seed for NextStat:

- PK (1-compartment oral) likelihood timing
- optional MLE fit wall-time (NextStat only)

This does **not** yet include nlmixr2/Torsten baselines; those are tracked as follow-up tasks in the Public Benchmarks epic.

Run a single case:

```bash
python run.py --deterministic --out ../../out/pharma_pk_1c_oral.json
```

Suite runner (multiple generated cases):

```bash
python suite.py --deterministic --out-dir ../../out/pharma
```

