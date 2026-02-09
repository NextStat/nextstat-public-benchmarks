# Template only. In the public benchmarks repo, this should be implemented with:
# - a pinned CUDA base image
# - pinned driver/runtime version reporting in the baseline manifest
# - pinned NextStat wheel installation
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ca-certificates \
    git \
    openssl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /bench

COPY env/python/requirements.txt env/python/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    pip install -r env/python/requirements.txt

# Optional: install JAX (CUDA) for ML suite profiling.
# Template: requires driver + compatible CUDA runtime on the host.
ARG INSTALL_JAX_CUDA=0
COPY env/python/requirements-ml-jax-cuda12.txt env/python/requirements-ml-jax-cuda12.txt
RUN if [ "${INSTALL_JAX_CUDA}" = "1" ]; then pip install -r env/python/requirements-ml-jax-cuda12.txt; else echo "NOTE: INSTALL_JAX_CUDA=0 (skipping JAX CUDA)"; fi

ARG NEXTSTAT_WHEEL=""
COPY ${NEXTSTAT_WHEEL} /tmp/nextstat.whl
RUN if [ -f /tmp/nextstat.whl ]; then pip install /tmp/nextstat.whl; else echo "NOTE: NEXTSTAT_WHEEL not provided"; fi

COPY suites/ suites/
COPY manifests/ manifests/

CMD ["python3", "suites/hep/run.py", "--deterministic", "--out", "out/hep_simple_nll.json"]
