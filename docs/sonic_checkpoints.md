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

## Optional original-release conversion

```bash
orcs-sonic-checkpoint --version release --convert --smpl

# Reuse a local original-release file; keep output separate.
orcs-sonic-checkpoint --version release --ckpt /data/original/last.pt \
  --convert --smpl --out-dir /data/experiments/sonic-release-port
```

The command locates `scripts/port_sonic_checkpoint.py` in the installed `mocke`
checkout without importing `mocke`, then runs it with explicit input/output
paths. Missing conversion code fails with setup guidance; no dependency is
modified automatically. The existing converter's checks remain in force.
Conversion subprocesses run with Hugging Face offline mode enabled.

Default output:

```text
$ORCS_DEPS_ROOT/checkpoints/sonic/release/ported/
├── last_ported.pt
├── smpl_ported.pt       # only with --smpl
└── conversion.json
```

An existing output or conversion lock is never overwritten. Use a new output
directory for another run. Failed conversions do not publish partial model
files. `conversion.json` records source and converter checksums plus output
checksums. An interrupted process may leave `.convert.lock`; inspect the directory
and confirm no converter is running before manually recovering it.

`--smpl` and `--out-dir` require `--convert`. Requests to convert `v1.1` or
`low-latency` are rejected before fetching weights or writing outputs: their
observation/window/runtime contracts are not yet adapted. No force bypass is
provided. The script does not fetch a planner or ONNX deployment models.

The equivalent checkout entry point is:

```bash
bash scripts/setup/sonic_checkpoint.sh --version v1.1 --local-files-only
```

This feature does not change `core/rl.py` or make existing tasks automatically
load these outputs. Runtime model selection and compatibility validation remain
separate work. Local tests use fake weights/cache and do not certify any real
checkpoint conversion or hardware behavior.
