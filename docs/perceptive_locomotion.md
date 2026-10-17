# perceptive locomotion

PerLoco pairs each privileged motion with its terrain tile. The AdaptSonic
policy receives a terrain height scan; the frozen SONIC base and motion command
remain shared with the other ORCS tasks.

## prerequisites

Complete the root [installation](../readme.md#setup) with the `perloco,smpl` extras
inside an active Python 3.11 environment. You also need Git LFS and enough disk
space for the selected source data.

GRAIL SMPL staging requires the licensed **SMPL-X v1.1 NPZ** release. Register
at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de), download
`SMPL-X v1.1 (NPZ+PKL)`, and keep `SMPLX_NEUTRAL.npz` ready. The setup script
will print the exact destination and pause until the file exists — or, with your
login in `SMPLX_USER` / `SMPLX_PASS`, download it for you. Stage the
male and female `.npz` alongside it: reconstructed UOLM staging wants SMPL-H
(`ORCS_SMPLH_DIR`, `smplh/SMPLH_*.pkl`) and falls back to `SMPLX_MALE.npz`.
The default SMPL-X root is `$ORCS_DEPS_ROOT/body_models`; no GRAIL code checkout
is required. `ORCS_SMPLX_DIR` overrides it.

## fetch and stage GRAIL

From the ORCS repository root:

```bash
bash scripts/setup/perceptive_locomotion.sh --sources grail
```

The command is idempotent and resumable. It:

1. downloads only the GRAIL curb families named by the packaged roster;
2. downloads `SMPLX_NEUTRAL.npz` if `SMPLX_USER`/`SMPLX_PASS` are set, else waits
   for it at the printed path (SMPL only);
3. stages terrain, robot motion, and SMPL motion under
   `$ORCS_DATA_ROOT/terrain_motions/grail`.

The current roster stages 63 clips over 8 curb tiles. Source selection lives in
`src/orcs/tasks/perloco/rosters/grail.toml`; the same roster controls download,
staging, retargeting, and runtime sampling.

To fetch both supported PerLoco datasets instead:

```bash
bash scripts/setup/perceptive_locomotion.sh
```

Use `--no-smpl` when you do not need the `-Smpl` task; it also skips GRAIL's
`recon/` data and the licensed body-model gate. Run
`bash scripts/setup/perceptive_locomotion.sh --help` for fetch-only,
stage-only, and non-interactive options.

## verify the staged task

```bash
python -c "import mjlab, orcs; print(orcs.tasks.perloco.SKIP_REASON)"
play Orcs-PerLoco-Grail-AdaptSonic --agent initial --viewer native
```

`SKIP_REASON == {}` means every staged PerLoco task that has its required data
registered successfully. Before kinematic retargeting, a skip reason for the
`-Smpl` task is expected because `seed_state.npz` does not exist yet.

Inspect terrain-motion alignment in the browser viewer:

```bash
orcs-view-terrain --source grail
# opens http://localhost:8080
```

## train

```bash
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-OmRe-AdaptSonic --env.scene.num-envs 4096
```

For SMPL point tracking and seed-backed RSI, continue with
[SMPL retargeting](smpl_retargeting.md).

## troubleshooting

- `No module named pxr` or `smplx`: rerun `scripts/setup/sync_deps.sh` (without
  `--no-smpl`) so the tooling and pinned `rsl_rl` fork are installed in the
  final install.
- An LFS pointer is being parsed as data: install Git LFS and rerun the setup
  script; do not run an unscoped `git lfs pull` in the full GRAIL dataset.
- A task is absent: read `orcs.tasks.perloco.SKIP_REASON`; registration skips
  missing datasets instead of making `import orcs` fail.

## GRAIL stairs (server-side)

Stairs use independent selectors, staged data and tasks; the default curb setup
is unchanged. `stair1`/`stair2` are aliases for the published directories
`stair_p1`/`stair_p2`. Do not combine curb and stairs in one invocation.

Activate the project environment installed with the `perloco,smpl` extras.
Keep data and checkpoints on the server, then run:

```bash
export ORCS_DATA_ROOT=/data/orcs/datasets
export ORCS_DEPS_ROOT=/data/orcs/dependencies
bash scripts/setup/perceptive_locomotion.sh \
  --sources grail --grail-categories stair1,stair2 --yes
```

The packaged `rosters/grail_stairs.toml` selects the first two complete families
per category, two takes each, in lexical order. Completeness means that robot,
reconstruction, object, metadata and USD files exist in the pinned Git tree;
it is NOT a quality or ascending-motion label. Inspect the selected videos or
replays before training. Use `--with-video` to include only selected videos.

The first resolution writes `$ORCS_DATA_ROOT/GRAIL-stairs-selection.json`, with
an immutable source commit, exact sample IDs and dependency paths. Later runs
reuse it, even when upstream changes. `--roster /path/selector.toml` accepts an
explicit `[[samples]]` list (category and stem) or selection limits; set
`all_samples = true` only when intentionally fetching the full selected categories.
Changing the selector requires a new `--selection /path/receipt.json`.
A fixed `revision = "<commit SHA>"` can be supplied in the selector.

Raw payloads go into `GRAIL-stairs/`; converted data goes into
`terrain_motions/grail_stairs/`. Each take owns a namespaced terrain tile so
variants cannot silently share a differently scaled/placed asset. `roster.toml`
in the output selects only the current receipt, not old tiles left on disk.
`stage_report.json` lists ready, cached and failed samples. A failed conversion
blocks stair task registration; inspect the named sample instead of training on
an accidental partial corpus.

`--no-smpl` skips reconstruction payloads and licensed model checks. `--skip-clone`
requires the pinned receipt and complete local files. `--skip-stage` fetches only;
it does not imply that any new task is ready. No seeds are generated by setup.

The converter bakes USD transforms/units/up-axis and explicit static object
pose/scale. Regular solid boxes remain boxes; other watertight surfaces use
CoACD convex pieces (never one whole-stair hull). Dynamic assets, ambiguous pose
schemas, unsupported instances and invalid meshes fail with a sample-specific
error. Geometry is approximate after convex decomposition: compare tread edges
and contact surfaces against the original USD in server acceptance tests.
The existing GRAIL phase-retiming convention is retained and recorded; endpoint
checks are only a gross alignment guard, not a replacement for replay inspection.

```bash
orcs-view-terrain --source grail_stairs
play Orcs-PerLoco-GrailStairs-AdaptSonic --agent initial --viewer native
orcs-pseudo-retarget --scene perloco-grail-stairs --all
# Replace <tile> with a tile ID from stage_report.json.
orcs-view-seeds --scene perloco-grail-stairs \
  --source /data/orcs/datasets/terrain_motions/grail_stairs/<tile>/level_0.00/sample0
play Orcs-PerLoco-GrailStairs-AdaptSonic-Smpl --agent initial --viewer native
train Orcs-PerLoco-GrailStairs-AdaptSonic-Smpl --env.scene.num-envs 64
```

The SMPL task is expected to skip until matching valid seeds exist. Seeds carry
source-content receipts; changed motions/metadata/terrain invalidate them. A
changed conversion publishes a new tile without retaining the previous seed.
Both tasks use the existing ORCS SONIC checkpoints and height-scan observations;
this does NOT add SONIC v1.1 support or a depth-vision foothold controller.

Before scaling up, verify a small selection from both categories: human/robot/
terrain alignment, collision surfaces, initialization, reset/step, and seed
playback. Do not start with 4096 environments: general meshes and larger grids
have different memory costs. Local contract tests do not establish real-data
conversion, GPU physics performance or policy convergence.
