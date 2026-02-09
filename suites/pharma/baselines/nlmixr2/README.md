# nlmixr2 Baseline (Template)

This directory holds an **R baseline runner** for the pharma suite using `nlmixr2`.

Status: template. The current environment does not include `nlmixr2`, so the runner will emit a machine-readable `skipped` result until `nlmixr2` is installed/pinned.

Run:

```bash
Rscript run.R --out out/nlmixr2_theoph_fit.json
```

Pinning guidance (planned):

- Use a pinned R toolchain + `renv.lock` (or a Docker image) for reproducible installs.
- Record `sessionInfo()` and package versions in the output JSON.

