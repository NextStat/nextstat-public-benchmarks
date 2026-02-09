FROM python:3.13-slim

WORKDIR /bench

# System deps for common Python wheels and crypto tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    openssl \
  && rm -rf /var/lib/apt/lists/*

COPY env/python/requirements.txt env/python/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install -r env/python/requirements.txt

# Optional: install JAX (CPU) for ML suite profiling.
ARG INSTALL_JAX_CPU=0
COPY env/python/requirements-ml-jax-cpu.txt env/python/requirements-ml-jax-cpu.txt
RUN if [ "${INSTALL_JAX_CPU}" = "1" ]; then pip install -r env/python/requirements-ml-jax-cpu.txt; else echo "NOTE: INSTALL_JAX_CPU=0 (skipping JAX)"; fi

# Install NextStat wheel (provided by CI/release artifact).
# Example:
#   docker build --build-arg NEXTSTAT_WHEEL=dist/nextstat-0.1.0-*.whl -f env/docker/cpu.Dockerfile .
ARG NEXTSTAT_WHEEL=""
COPY ${NEXTSTAT_WHEEL} /tmp/nextstat.whl
RUN if [ -f /tmp/nextstat.whl ]; then pip install /tmp/nextstat.whl; else echo "NOTE: NEXTSTAT_WHEEL not provided"; fi

COPY suites/ suites/
COPY manifests/ manifests/

CMD ["python", "suites/hep/run.py", "--deterministic", "--out", "out/hep_simple_nll.json"]
