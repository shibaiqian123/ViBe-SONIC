"""Transactional stair staging; failures never publish a partial sample."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def converter_digest() -> str:
    from orcs.cli import stage_terrain_motions
    from orcs.tasks.perloco.sources import grail_stairs, smplx_fk, stair_geometry
    h = hashlib.sha256()
    for module in (grail_stairs, stair_geometry, smplx_fk, stage_terrain_motions):
        h.update(Path(module.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def stage_stairs(src, out_root: Path, fps: float, batch: int, device: str) -> None:
    import torch

    from orcs.cli._motion_npz import Fk
    from orcs.cli.stage_terrain_motions import (
        _resample,
        _resample_smpl,
        _velocities,
        _write_clip,
        _write_tile,
    )
    from orcs.core.paths import REPO_ROOT, smplx_dir
    from orcs.tasks.perloco.sources.grail_stairs import fingerprint

    out_root.mkdir(parents=True, exist_ok=True)
    report = {"status": "staging", "revision": src.selection["revision"], "ready": [], "failed": {}, "cached": []}
    fk = None
    converter = converter_digest()
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None  # A wheel has no Git checkout; the source digest remains authoritative.
    if src.smpl:
        models = sorted((smplx_dir() / "smplx").glob("SMPLX_*.npz"))
        if not models:
            raise FileNotFoundError("SMPL-X body models are missing")
        converter += fingerprint(models)
    # All source input paths were already validated by the fetch receipt.
    for row in src.samples:
        target = out_root / row["tile_id"] / "level_0.00"
        try:
            src._load(row)
            tile, clip = src._tiles[row["tile_id"]], src._clips[row["tile_id"]]
            config = {"converter_sha256": converter, "fps": fps, "smpl": src.smpl,
                      "input_sha256": clip.meta["input_sha256"]}
            token = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            old = target / "sample0" / "metadata.json"
            try:
                cache_matches = old.exists() and json.loads(old.read_text()).get("stage_fingerprint") == token
                if cache_matches:
                    import numpy as np
                    with np.load(target / "sample0" / "motion.npz", allow_pickle=False) as motion:
                        frames = len(motion["joint_pos"])
                        cache_matches = frames >= 3 and np.isfinite(motion["joint_pos"]).all()
                    json.loads((target / "tile.json").read_text())
                    if src.smpl:
                        with np.load(target / "sample0" / "smpl_motion.npz", allow_pickle=False) as human:
                            cache_matches = cache_matches and len(human["smpl_joints"]) == frames
                if cache_matches:
                    report["cached"].append(row["tile_id"])
                    continue
            except (OSError, ValueError, KeyError):
                pass  # A corrupt cache is regenerated, never accepted as a valid hit.
            if fk is None:
                fk = Fk(batch, fps, device)
            perm = [list(clip.joint_names).index(n) for n in fk.il_names]
            state, phase = _resample(clip, fps, device)
            if len(phase) < 3:
                raise ValueError("clip too short after resampling")
            state["joint_pos"] = state["joint_pos"][:, perm]
            vel = _velocities(state, 1.0 / fps)
            bodies = fk(state, vel)
            human = _resample_smpl(clip.smpl, phase, device) if clip.smpl else None
            if not all(torch.isfinite(x).all() for x in (*state.values(), *vel.values())):
                raise ValueError("nonfinite resampled motion")
            clip.meta.update(stage_fingerprint=token, converter_sha256=converter, generator_sha=commit)
            with tempfile.TemporaryDirectory(prefix=".stair-stage-", dir=out_root) as tmp:
                temporary_root = Path(tmp)
                _write_tile(temporary_root, tile)
                staged = temporary_root / clip.tile_key
                _write_clip(staged / "sample0", clip, state, vel, bodies, fps, fk.il_names, human)
                # Preserve the previous complete tile until the replacement is ready.
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = target.with_name(".previous")
                if backup.exists():
                    raise ValueError(f"interrupted publication; inspect {backup}")
                if target.exists():
                    os.replace(target, backup)
                try:
                    os.replace(staged, target)
                except BaseException:
                    if backup.exists():
                        os.replace(backup, target)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
            report["ready"].append(row["tile_id"])
            print(f"[stairs stage] {row['stem']} -> {target}", flush=True)
        except Exception as exc:
            report["failed"][row["tile_id"]] = f"{type(exc).__name__}: {exc}"
            print(f"[stairs ERROR] {row['stem']}: {exc}", flush=True)
    report["status"] = "failed" if report["failed"] else "publishing"
    (out_root / "stage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["failed"]:
        raise RuntimeError(f"{len(report['failed'])} stair samples failed; see {out_root / 'stage_report.json'}")
    # Runtime whitelist follows the receipt, not every old tile left on disk.
    families = [row["tile_id"] for row in src.samples]
    (out_root / "roster.toml").write_text("families = " + json.dumps(families) + "\nlevels = [0.0]\n")
    report["status"] = "complete"
    (out_root / "stage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[stairs stage] ready={len(report['ready'])}, cached={len(report['cached'])}")
