# Data

The preferred source is the ROBUST-MIPS Synapse release:

- Synapse entity: `syn64023381`
- DOI: <https://doi.org/10.7303/syn64023381>
- Reported licence: CC BY-NC-SA

ROBUST-MIPS is expected to contain the selected raw images, corrected instrument-instance masks, and pose JSON annotations. The original ROBUST-MIS video data should not be downloaded unless the ROBUST-MIPS integrity audit shows it is required.

Release v1 names the per-frame pose annotation `toolposes.json` (rather than
`raw.json`, as shown in the paper's directory schematic). The archive also
contains macOS AppleDouble files under `__MACOSX`; these are ignored.

Expected local-only directories:

- `data/archives/`
- `data/raw/`

Checksums, entity versions, archive structure, access conditions, and exact counts will be recorded here after acquisition.

## Local acquisition status (2026-09-25)

- Archive: `data/archives/RobustMIPS.zip`
- Synapse file: `syn68915165`, version 1
- Size: 3,533,771,766 bytes
- MD5: `072fd8e9b7d73c66f12039c3cc2d5ac2`
- SHA-256: `bb1e5a2ee4ad3cadd8bb1b3f5d4222703b30615c8456a908fd48056c7531f832`
- Extracted payload: `data/raw/RobustMIPS/` (approximately 3.5 GiB on disk)
- Extracted counts: 10,040 `raw.png`, 10,040 `instrument_instances.png`, and 10,040 `toolposes.json`
- Packaging metadata (`__MACOSX`, AppleDouble files, `.DS_Store`, and `Thumbs.db`) was excluded during extraction.

The machine-readable C1 results are under `data/manifests/c1_audit/`.

C2 normalized frame, pose-instance, mask-instance, anomaly, and schema
manifests are under `data/manifests/c2/`. See its README for the important
pose-order correspondence caveat.
