# SONIC checkpoint versions and cache

`orcs-sonic-checkpoint` locates official weights without loading a model,
importing MJLab/PyTorch, registering tasks, or changing training configuration.
Run acquisition and conversion on the server; keep only code and configuration
in your local Git checkout. Install/update ORCS dependencies first.

Available versions in `nvidia/GEAR-SONIC`:

- `release` (default): `sonic_release/last.pt`
- `v1.1`: `sonic_v1_1/last.pt`
- `low-latency`: `low_latency/last.pt`

## Existing files and cache

```bash
# Existing server file: no download, no conversion.
orcs-sonic-checkpoint --version v1.1 --ckpt /actual/path/sonic_v1_1/last.pt

# Search only the server's configured Hugging Face cache; fail on a miss.
orcs-sonic-checkpoint --version v1.1 --local-files-only

# Explicit cache and source commit; downloads only when the SDK needs it.
orcs-sonic-checkpoint --version low-latency \
  --cache-dir /data/huggingface/hub --revision <commit-sha>
```

An explicit `--ckpt` always wins. A missing/empty file or Git LFS pointer fails;
there is no fallback to a different version. A local file's version is declared
by the user, not inferred or verified from its contents. Hub cache/revision
options do not validate or download anything when `--ckpt` is supplied.

Without `--ckpt`, the command calls `hf_hub_download` for the selected filename.
Omitted `--revision` and `--cache-dir` preserve the SDK defaults, including its
environment-based cache configuration. Normal mode may contact the Hub to resolve
a revision even with cached weights; use `--local-files-only` to prohibit that.
The cached file is used directly, without another large checkpoint copy.

Every successful resolution prints the actual path and writes a small,
content-addressed JSON source receipt under:

```text
$ORCS_DEPS_ROOT/checkpoints/sonic/<version>/sources/
```

Receipts record the source path, SHA-256, size, selected version and actual Hub
snapshot commit (or user-declared local provenance). They never claim that
resolution validates model semantics or training compatibility. Resolving weights
therefore requires write access for receipts, even in offline mode.

## Three-variant stair integration

The native ORCS converter (`core/sonic/port.py`) supports all three variants. It pins
snapshot `6733128a3d8a523b1418b06bca3cdf61c8b0987f` by default and verifies the
published SHA-256 of each training checkpoint before loading. Unknown custom
weights are rejected, even if their shapes match. mocke/rsl_rl are unchanged.

```bash
orcs-sonic-checkpoint --version release --convert --smpl
orcs-sonic-checkpoint --version v1.1 --convert --smpl
orcs-sonic-checkpoint --version low-latency --convert --smpl
```

Outputs remain version-isolated; `parity.json` records source/remapped network
checks. Layer discovery is dynamic and unsupported control parameters fail
closed. Base encoder and decoder weights stay frozen; only decoder adapters train.
The v1.1 reference orientation is robot-heading-relative. Low latency uses a
10-frame G1 window at 0.02 s and 4-frame SMPL window at 0.02 s. All variants retain
10-frame proprioception and 50 Hz control. Changes are opt-in to stair tasks only.

New task suffixes are `-V11`, `-V11-Smpl`, `-LowLatency`, and `-LowLatency-Smpl`
on `Orcs-PerLoco-GrailStairs-AdaptSonic`. Existing task names still select release,
with legacy mocke release checkpoint fallback. Resume rejects different version,
mode or base-weight fingerprints. An untagged legacy release training checkpoint
is accepted only when every frozen parameter matches the selected base exactly.

```bash
orcs-pseudo-retarget --scene perloco-grail-stairs --sonic-version v1.1 --all
orcs-view-seeds --scene perloco-grail-stairs --sonic-version v1.1 --source /actual/sample
play Orcs-PerLoco-GrailStairs-AdaptSonic-V11-Smpl --agent initial
train Orcs-PerLoco-GrailStairs-AdaptSonic-V11-Smpl --env.scene.num-envs 64
```

Release seeds keep `seed_state.npz`; other variants use
`seeds/<version>/seed_state.npz`. Receipts include model provenance. Old stair
seeds lacking model provenance must be regenerated; other tasks are unaffected.

`tests/verify_official_sonic_onnx.py` compares the cached official ONNX graphs to
the ORCS remapping on CPU. It does not create training weights or substitute for
last.pt verification. The pinned low-latency ONNX actually packs single-frame
orientation before multi-frame orientation, unlike its published observation YAML;
the verifier reports and respects the graph layout. ORCS constructs branch-level
features directly and does not use that all-mode deployment packing.

Real reset/step and short PPO acceptance still require staged stairs, matching
seeds and a complete MJLab/assets environment. Numerical parity is not evidence
of terrain success, convergence or hardware readiness.

Conversion runs in a temporary output directory and publishes only after all
checks pass. Existing results are not overwritten. `--smpl` and `--out-dir`
require `--convert`. A missing explicit `--ckpt` never falls back to downloading.
The shell entry point remains `bash scripts/setup/sonic_checkpoint.sh`.

See the [dated validation record](validation/sonic_variants_2026-10-17.md) for
actual numerical results and the remaining physics/PPO acceptance work.
