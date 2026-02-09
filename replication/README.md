# Third-Party Replication (Template)

Goal: enable an independent party to rerun a published benchmark snapshot and produce a **signed replication report**.

This folder is a template; the actual replication requires an external runner and their signature material.

## Inputs

- A published snapshot artifact set (zip/tar) containing:
  - `baseline_manifest.json`
  - `snapshot_index.json`
  - `nextstat_wheel.whl` (recommended: the exact wheel used for the run)
  - suite outputs (raw JSONs)
  - `README_snippet.md`
- The harness repository at the recorded commit SHA.

## Replication Steps (Minimal)

1. Obtain the snapshot id and baseline manifest from the published artifacts.
2. Check out the harness at `baseline_manifest.harness.git_commit`.
3. Recreate the pinned environment (Python/Rust/Docker as documented in the harness).
4. Rerun suites using the same flags and deterministic policy.
5. Produce a new baseline manifest for the rerun environment and hash all artifacts.
6. Write a rerun `snapshot_index.json` and generate a `replication_report.json` by comparing snapshot indices.
7. Compare (semantically, beyond hashes):
   - schema validity
   - dataset ids + hashes
   - numerical correctness gates
   - timing deltas (report distributions, not just a single number)
8. Fill out `signed_report_template.md` and sign it (GPG/minisign/etc).

## Seed Tooling

- `compare_snapshots.py` compares two snapshot directories and emits a structured JSON diff (datasets + hashes + suite-specific checks where supported).
- `make_rerun_bundle.py` creates a rerun snapshot and writes `snapshot_comparison.json` + `signed_report_draft.md`.
- `scripts/write_snapshot_index.py` + `scripts/write_replication_report.py` provide schema-backed hash inventories and a machine-readable replication report.

## Output

- `signed_report.md` (filled template)
- `signed_report.md.sig` (signature)
- rerun artifact set (baseline manifest + raw results)
