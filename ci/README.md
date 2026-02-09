# CI Workflows (Template)

This folder contains GitHub Actions workflow templates for the standalone public benchmarks repo.

Important: GitHub Actions only runs workflows from `.github/workflows/`.

Two options:

1) During export from the monorepo seed, use:

```bash
python3 benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
  --out /path/to/nextstat-public-benchmarks \
  --with-github-workflows
```

2) Or manually copy the templates in your standalone repo:

```bash
mkdir -p .github/workflows
cp ci/*.yml .github/workflows/
```

Workflows:

- `verify.yml`: smoke verifies the harness with a pinned NextStat wheel from repo variables
- `publish.yml`: manually publishes a snapshot artifact directory (baseline manifest + index + raw results)
- `publish_gpu.yml`: publishes a snapshot on a self-hosted GPU runner (intended for ML `jax_jit_gpu_*` cases)

GPU notes:

- `publish_gpu.yml` expects a runner with CUDA drivers and (optionally) `nvidia-smi`.
- Runner selection is controlled by the `runner_labels_json` workflow input (JSON array of labels).
