# GRAIL focus audit — 2026-10-18

Status: **blocked; not approved for SMPL training**. No formal dataset or seed was overwritten.

## Reproducibility

- Sample: `terrain_stairs__stairs_2065__0001`, release revision `40e795761302e611c1e7e3a6caefdd010d56c199`.
- NVIDIA upstream commit: `da382dcac447261456314c3e3ce15401d5dc1eb0`.
- Official GMR submodule commit: `bb1bbe40774794fceb2a7c579a3464a28e68c844` (not executed; no IK in this audit).
- Independent server code: `/home/cqu023/orcs-grail-official-audit/upstream-sparse`.
- Final numeric report: `/home/cqu023/orcs-grail-official-audit/official-focus-geometry/report.json`.
- Arrays remain alongside that report on the server.
- `/home/cqu023/GRAIL` is a personal fork at `b760fee4f7284bc668668b167d888c003c7c8bad`, not the official baseline. Initial `focus-v1` output is superseded by the official audit.

## Findings

The published robot sequence contains 250 frames at 25 Hz (9.96 seconds between endpoints).
Reconstruction contains 241 frames at 30 Hz (8 seconds). The official exporter writes zero
placeholder `smpl_joints`; they cannot supply correspondence. No source frame mapping has been
established. Current endpoint-normalized pairing is an approximation, not an upstream contract.
Frames 0/228/286/436 therefore have no certified corresponding reconstruction frames.

Running the pinned official `forward_smplx` function on unpadded reconstruction frames yields:

- Standard model with authored scale: maximum ORCS joint difference **0 m**.
- Official G1-template branch (authored scale disabled as upstream specifies): maximum difference
  **0.069605231 m**, mean **0.024711018 m**. This is a model-branch difference, not proof that the
  G1 branch should replace the current source interpretation. Release metadata does not identify the branch.

Published robot trajectory versus staged robot channels passes the proposed tolerances:

- Maximum pelvis position difference: **5.364418e-7 m**.
- Maximum root orientation difference: **0.07403773 degrees**.
- Maximum joint difference: **0.000189538 degrees**.

This uses the existing named ORCS joint-order map, not an independent official robot FK.

Reproducing the official renderer's root translate/orient/scale replacement in an anonymous USD
session layer and comparing with ORCS static geometry gives:

- First-frame maximum vertex difference: **1.022932e-6 m**.
- Whole robot sequence maximum difference: **0.001753027 m**, exceeding the 1 mm gate.

The second number reflects authored object motion versus ORCS's first-frame static approximation.
No geometry was shifted to hide it. This millimetre-level mismatch does not establish the cause
of the earlier much larger hovering. Convex-decomposition collision surfaces remain unverified.

## Validation limits and next evidence needed

The geometric audit viewer started and ran its update loop for 20 seconds on port 8093 without
an exception. A browser visual inspection and complete official Isaac Sim MP4 are not completed.
Isaac Sim exists in the server `sonic` environment, but a rendered video is not claimed here.

The focus sample fails overall acceptance, so the remaining seven samples and new seed generation
were not run. Do not rerun seed with the same configuration to resolve missing provenance.
Required next evidence: recon-to-exported-rollout time mapping and the human-template configuration
used to produce this release; otherwise the paired SMPL route remains blocked. Existing robot-only
references can still be inspected separately.
