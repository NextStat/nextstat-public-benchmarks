# Python Environment (Pinned)

This folder pins the Python dependencies used by the benchmark harness.

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r env/python/requirements.txt
```

NextStat is intentionally **not** pinned here: snapshot manifests must record the exact NextStat version / wheel hash used for the run.

## Optional: ML suite (JAX)

The ML suite can include JAX JIT backends:

- `jax_jit_cpu_*` (CPU)
- `jax_jit_gpu_*` (CUDA; requires GPU runner)

CPU install (pinned template):

```bash
pip install -r env/python/requirements-ml-jax-cpu.txt
```

CUDA install (template):

```bash
pip install -r env/python/requirements-ml-jax-cuda12.txt
```

Notes:
- These pins are a **starting point**; if your Python/CUDA combo differs, update them.
- GPU runs should be performed in a pinned CUDA container (`env/docker/cuda.Dockerfile`) or on a self-hosted runner with a declared driver/runtime.
