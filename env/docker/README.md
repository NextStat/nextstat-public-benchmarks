# Docker Environments (Templates)

These Dockerfiles are **templates** for running benchmark snapshots in a pinned environment.

- `cpu.Dockerfile`: CPU-only runs
- `cuda.Dockerfile`: CUDA runs (optional, when GPU benchmarks are published)

The intent is:

1. Pin OS base image + Python version
2. Install pinned harness deps (`env/python/requirements.txt`)
3. Install a specific NextStat wheel (artifact under test)
4. Run suite(s) and publish artifacts + manifests

## ML suite (JAX) toggles

Both templates include optional build args for JAX:

- CPU: `--build-arg INSTALL_JAX_CPU=1` (installs `env/python/requirements-ml-jax-cpu.txt`)
- CUDA: `--build-arg INSTALL_JAX_CUDA=1` (installs `env/python/requirements-ml-jax-cuda12.txt`)

Example (CUDA, ML suite):

```bash
docker build \
  -f env/docker/cuda.Dockerfile \
  --build-arg NEXTSTAT_WHEEL=tmp/nextstat.whl \
  --build-arg INSTALL_JAX_CUDA=1 \
  -t nextstat-public-bench:cuda .
docker run --rm --gpus all nextstat-public-bench:cuda \
  python3 scripts/publish_snapshot.py --snapshot-id snapshot-gpu --deterministic --ml
```
