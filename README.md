# NextStat Public Benchmarks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18542624.svg)](https://doi.org/10.5281/zenodo.18542624)

Reproducible benchmark harness for NextStat — pinned environments, schema-validated artifacts, DOI-ready snapshots.

Goal: make benchmark snapshots **rerunnable by outsiders** with pinned environments, correctness gates, and raw artifact publishing.

Canonical benchmark program spec (in `nextstat.io`): `docs/benchmarks/public-benchmarks.md`.

## Quickstart (local)

1. Create a Python venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install pinned harness deps:

```bash
pip install -r env/python/requirements.txt
```

3. Install NextStat:

- If you have a wheel:
  - `pip install /path/to/nextstat-*.whl`
- If you are running from the `nextstat.io` monorepo:
  - build a wheel via `maturin build --release` in `bindings/ns-py`, then `pip install target/wheels/*.whl`.

4. Run the minimal HEP suite (NLL parity + timing):

```bash
python suites/hep/run.py --deterministic --out out/hep_simple_nll.json
```

5. Run the Bayesian suite (NUTS diagnostics + ESS/sec proxies):

```bash
python suites/bayesian/suite.py --deterministic --out-dir out/bayesian
```

6. Run the ML suite (compile vs execution, seed):

```bash
python suites/ml/suite.py --deterministic --out-dir out/ml
```

Optional: enable JAX backends for ML suite (CPU):

```bash
pip install -r env/python/requirements-ml-jax-cpu.txt
```

Optional: enable JAX backends for ML suite (CUDA, on a GPU runner):

```bash
pip install -r env/python/requirements-ml-jax-cuda12.txt
```

Optional: also benchmark full MLE fits (more expensive):

```bash
python suites/hep/run.py --deterministic --fit --fit-repeat 3 --out out/hep_simple_nll_fit.json
```

## Export to a standalone repo (from the monorepo seed)

If you are starting from the `nextstat.io` monorepo and want a clean standalone repo directory
(excluding local outputs like `out/`, `tmp/`, `manifests/snapshots/`), use:

```bash
# From the monorepo root:
python3 benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
  --out /path/to/nextstat-public-benchmarks
```

Optional: also stage GitHub Actions workflows into `.github/workflows/` in the exported repo:

```bash
python3 benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
  --out /path/to/nextstat-public-benchmarks \
  --with-github-workflows
```

Then:

```bash
cd /path/to/nextstat-public-benchmarks
git init
git add -A
git commit -m "Initial public benchmarks harness"
```

## CI configuration (template)

The workflow templates under `ci/` expect a **pinned** NextStat wheel to be installed (so published snapshots can record the exact build being measured).

- In a standalone GitHub repo, copy these files into `.github/workflows/` (GitHub Actions only runs workflows from that folder).
- The wheel **must** match the runner OS/arch and Python version. The templates use `ubuntu-latest` + Python `3.13` by default.
- `ci/verify.yml` (PR/push): set GitHub Actions variables:
  - `NEXTSTAT_WHEEL_URL` — URL to the wheel file
  - `NEXTSTAT_WHEEL_SHA256` — SHA-256 of the wheel file (hex)
- `ci/publish.yml` (manual): either:
  - provide `nextstat_wheel_url` + `nextstat_wheel_sha256`, or
  - leave them empty and provide `nextstat_ref` to build the wheel from source (optionally override `nextstat_repo` / `nextstat_py_subdir`).
  You can also toggle `run_hep` / `run_pharma`.

### GPU publish workflow notes

`.github/workflows/publish_gpu.yml` runs on a **self-hosted GPU runner**. It requires a pinned NextStat wheel and supports two ways to provide it:

- `nextstat_wheel_url` + `nextstat_wheel_sha256` (download), or
- `nextstat_wheel_path` + `nextstat_wheel_sha256` (copy from runner-local filesystem).

Both publish workflows also generate a **DOI-ready package** (archive + SHA256 + `zenodo_deposition.json`) as a separate artifact named `doi-package-<snapshot_id>`.

## Publish A Local Snapshot (Seed)

Generate a local snapshot directory (suite outputs + `baseline_manifest.json` + `snapshot_index.json` + README snippet) under `manifests/snapshots/<snapshot_id>/`:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --fit --fit-repeat 3
```

By default this runs the `hep` and `pharma` suites. Add `--bayesian` to include the Bayesian suite:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --fit --fit-repeat 3 --bayesian
```

If you want the baseline manifest to pin the exact measured NextStat build, pass the wheel path so the manifest records `nextstat.wheel_sha256`:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --nextstat-wheel /path/to/nextstat-*.whl
```

When `--nextstat-wheel` is provided, the wheel is copied into the snapshot directory as `nextstat_wheel.whl` (so the snapshot can be self-contained for DOI publishing).

Suites:

- `hep/` (pyhf vs NextStat)
- `pharma/` (NextStat-only seed + baseline templates)
- `bayesian/` (NextStat-only seed: NUTS diagnostics + ESS/sec proxies; only when `--bayesian` is passed)
- `ml/` (NextStat-only seed: cold-start TTFR vs warm-call throughput; optional JAX cases)

## DOI Publishing (Template)

For Zenodo/DOI publishing guidance and metadata templates, see `zenodo/`.

## What This Seed Provides

- `env/` pinned environment scaffolding (Python + Rust + Docker templates)
- `CITATION.cff` citation metadata template (fill DOI/version on release)
- `manifests/schema/` JSON Schemas for results + baseline manifests
- `suites/` suite layout (currently runnable: `hep`, `pharma`, `bayesian`, `ml`)
- `ci/` workflow templates for verify/publish (standalone repo)

## Notes

- The harness is intentionally separate from the product repo: it should be auditable and runnable without building all of NextStat.
- Use `--json-only` in the validation-pack tooling when you want reproducibility artifacts without `matplotlib`/PDF.
