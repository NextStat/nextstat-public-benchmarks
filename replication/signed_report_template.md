# Third-Party Replication Report (Template)

## Snapshot Being Replicated

- Snapshot id:
- Published DOI (if any):
- Harness repo:
- Harness commit SHA:
- NextStat version / wheel hash (if provided):

## Replication Environment

- Runner name / org:
- Date:
- Machine:
  - OS:
  - CPU:
  - RAM:
  - GPU (if applicable):
- Toolchains:
  - Python:
  - Rust:
  - R / CmdStan (if applicable):

Attach the rerun baseline manifest:

- `rerun_baseline_manifest.json`:

## Artifact Integrity

- Published artifact set hash:
- Rerun artifact set hash:

## Schema Validity

- All JSON artifacts validated against published schemas: yes/no
- Notes:

## Correctness Gates

- HEP (NLL parity): pass/fail, worst case + abs/rel diff
- Pharma (sanity checks): pass/fail, notes
- Notes:

## Performance Deltas

Report distribution-aware comparisons (median + IQR or min + raw repeats):

- HEP NLL time/call:
- HEP fit wall-time:
- Pharma NLL time/call:
- Pharma fit wall-time:
- Notes on variability and system load:

## Conclusion

- Replication outcome: confirmed / not confirmed / partially confirmed
- Key differences:
- Follow-up required:

## Signature

- Signing method (GPG/minisign):
- Public key fingerprint:
- Signature file name:

