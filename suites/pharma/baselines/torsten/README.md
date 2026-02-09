# Torsten (Stan) Baseline (Template)

This directory is a **template** for running a Torsten (Stan) baseline for PK/NLME workflows.

Status: template. It emits a machine-readable `skipped` baseline result until:

- a pinned CmdStan toolchain is installed
- Torsten-enabled CmdStan build is available
- a declared fit protocol is implemented (optimizer, stopping rules, init)

Run:

```bash
Rscript run.R --out out/torsten_pk_fit.json
```

