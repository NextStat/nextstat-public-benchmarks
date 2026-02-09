# HEP Suite (Minimal Seed)

This is a minimal HEP benchmark seed: **NLL parity + timing** (and optional **MLE fit timing**) for a tiny HistFactory workspace.

Inputs:
- `datasets/simple_workspace.json` (tiny, in-repo fixture)

Outputs:
- A single JSON result following `manifests/schema/benchmark_result_v1.schema.json`

Run:

```bash
python run.py --deterministic --out ../../out/hep_simple_nll.json
```

Also benchmark full fits (more expensive):

```bash
python run.py --deterministic --fit --fit-repeat 3 --out ../../out/hep_simple_nll_fit.json
```

Compute profiled `q0`/`Z0` (conditional fit at `mu=0`):

```bash
python run.py --deterministic --profile --out ../../out/hep_simple_profile.json
```

Suite runner (multiple cases + synthetic scaling):

```bash
python suite.py --deterministic --out-dir ../../out/hep
```

With fits:

```bash
python suite.py --deterministic --fit --fit-repeat 3 --out-dir ../../out/hep_fit
```

With the optional ROOT/RooFit reference-path baseline (may be skipped if PyROOT isn't available):

```bash
python suite.py --deterministic --root-baseline --out-dir ../../out/hep_root
```
