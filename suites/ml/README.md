# ML Suite (Planned)

This directory is reserved for the **ML benchmark suite** in the standalone public benchmarks repo.

Canonical methodology + runbook lives in the main docs site:

- `/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/suites/ml.md`

Planned measurements (publishable snapshots):

- cold-start latency (time-to-first-result)
- warm throughput (steady-state execution)
- explicit cache policy (cold/warm cache modes)

Seed status (today):

- runnable seed: `suite.py` + `run.py` + `report.py`
- minimal backends:
  - `numpy` (always available via pinned harness deps)
  - `jax_jit_cpu` / `jax_jit_gpu` (optional; `warn` if JAX missing; GPU case fails if no GPU backend)

Optional: enable JAX backend (CPU, pinned template)

```bash
pip install -r env/python/requirements-ml-jax-cpu.txt
```

Offline/dev note (monorepo)

If you are running from the `nextstat.io` monorepo and cannot install pinned deps via `make install`
(e.g. no PyPI access), you can still run the ML suite seed with your existing venv:

```bash
./.venv/bin/python benchmarks/nextstat-public-benchmarks/suites/ml/suite.py \
  --deterministic --out-dir benchmarks/nextstat-public-benchmarks/out/ml
./.venv/bin/python benchmarks/nextstat-public-benchmarks/suites/ml/report.py \
  --suite benchmarks/nextstat-public-benchmarks/out/ml/ml_suite.json \
  --out benchmarks/nextstat-public-benchmarks/out/README_snippet_ml.md
```

Run:

```bash
python suites/ml/suite.py --deterministic --out-dir out/ml
python suites/ml/report.py --suite out/ml/ml_suite.json --out out/README_snippet_ml.md
```

This seed publishes component timings (import, first call, warm call distribution) under a pinned schema and is intended as a stable base for future JAX/torch.compile expansions.
