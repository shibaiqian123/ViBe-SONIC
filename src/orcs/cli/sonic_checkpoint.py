"""Locate official SONIC weights without loading models or discovering tasks.

Only the original release can currently be passed to mocke's porting script.
Resolving a filename is not proof of runtime compatibility or model identity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "nvidia/GEAR-SONIC"
VERSIONS = {
    "release": "sonic_release/last.pt",
    "v1.1": "sonic_v1_1/last.pt",
    "low-latency": "low_latency/last.pt",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", choices=tuple(VERSIONS), default="release")
    result.add_argument("--ckpt", type=Path, help="existing file; missing explicit paths never download")
    result.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    result.add_argument("--local-files-only", action="store_true", help="never access network for weights")
    result.add_argument("--revision", help="Hugging Face branch/tag/commit (SDK default when omitted)")
    result.add_argument("--convert", action="store_true", help="port original release through installed mocke")
    result.add_argument("--smpl", action="store_true", help="also port SMPL encoder; requires --convert")
    result.add_argument("--out-dir", type=Path, help="conversion directory; existing outputs are never replaced")
    return result


def validate_args(args: argparse.Namespace, cli: argparse.ArgumentParser) -> None:
    if args.convert and args.version != "release":
        cli.error(f"{args.version}: 当前转换器尚未验证兼容; only weight resolution is supported")
    if args.smpl and not args.convert:
        cli.error("--smpl requires --convert")
    if args.out_dir is not None and not args.convert:
        cli.error("--out-dir requires --convert")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_revision(path: Path) -> str | None:
    # Inspect the snapshot path BEFORE resolving its symlink into blobs/.
    for index, component in enumerate(path.parts[:-1]):
        if component == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", path.parts[index + 1]):
            return path.parts[index + 1].lower()
    return None


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.ckpt is not None:
        path = args.ckpt.expanduser().resolve()
        source = "local_file"
        revision = None
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install ORCS dependencies (huggingface-hub is required for cache/download lookup)") from exc
        kwargs = {"repo_id": REPOSITORY, "filename": VERSIONS[args.version],
                  "local_files_only": args.local_files_only}
        if args.cache_dir is not None:
            kwargs["cache_dir"] = str(args.cache_dir.expanduser())
        if args.revision is not None:
            kwargs["revision"] = args.revision
        cached_path = Path(hf_hub_download(**kwargs))
        revision = resolved_revision(cached_path)
        if revision is None:
            raise ValueError(f"Cannot establish resolved Hub commit from snapshot path: {cached_path}")
        path = cached_path.resolve()
        source = "huggingface_cache"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}; no fallback download")
    if path.stat().st_size == 0:
        raise ValueError(f"Checkpoint is empty: {path}")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Checkpoint is a Git LFS pointer, not weights: {path}")
    receipt = {
        "schema_version": 1,
        "selected_version": args.version,
        "version_identity": "user_declared_not_verified" if source == "local_file" else "official_repository_path",
        "source_kind": source,
        "repository": REPOSITORY if source != "local_file" else None,
        "repository_filename": VERSIONS[args.version] if source != "local_file" else None,
        "requested_revision": args.revision if source != "local_file" else None,
        "resolved_revision": revision,
        "checkpoint_path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "runtime_compatibility": "not_validated_by_resolution",
    }
    return path, receipt


def find_converter() -> Path:
    """Find the script in the installed editable checkout, never import mocke."""
    try:
        spec = importlib.util.find_spec("mocke")
    except (ImportError, ValueError):
        spec = None
    candidates = []
    if spec is not None:
        origins = [Path(spec.origin).parent] if spec.origin else []
        origins.extend(Path(p) for p in (spec.submodule_search_locations or []))
        for origin in origins:
            for parent in (origin, origin.parent, origin.parent.parent):
                script = parent / "scripts" / "port_sonic_checkpoint.py"
                if script.is_file():
                    candidates.append(script.resolve())
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            "Cannot locate one installed mocke scripts/port_sonic_checkpoint.py. "
            "Activate the ORCS project environment and run bash scripts/setup/sync_deps.sh; "
            "a wheel-only mocke install may not contain the script. No dependencies were changed."
        )
    return unique[0]


def preflight_output(out_dir: Path, smpl: bool) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"Not a directory: {out_dir}")
    # Refuse any prior conversion artifacts, including an unrequested SMPL file.
    for name in ("last_ported.pt", "smpl_ported.pt", "conversion.json", ".convert.lock"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to replace existing conversion: {out_dir / name}")


def save_source_receipt(root: Path, receipt: dict) -> Path:
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    identity = hashlib.sha256(payload.encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{identity}.json"
    try:
        with dest.open("x") as stream:
            stream.write(payload)
    except FileExistsError:
        if dest.read_text() != payload:
            raise ValueError(f"Conflicting source receipt: {dest}") from None
    return dest


def convert_checkpoint(script: Path, checkpoint: Path, out_dir: Path,
                       *, smpl: bool, receipt: dict) -> None:
    preflight_output(out_dir, smpl)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint disappeared before conversion: {checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".convert.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix=".conversion-", dir=out_dir) as temporary:
            temp = Path(temporary)
            output = temp / "last_ported.pt"
            command = [sys.executable, str(script), "--ckpt", str(checkpoint), "--out", str(output)]
            if smpl:
                command.append("--smpl")
            # Resolution is complete. The converter must never invoke its legacy downloader.
            subprocess.run(command, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
            names = ["last_ported.pt"] + (["smpl_ported.pt"] if smpl else [])
            for name in names:
                if not (temp / name).is_file() or (temp / name).stat().st_size == 0:
                    raise RuntimeError(f"Converter did not produce a nonempty {name}")
            conversion = {"source": receipt, "converter_path": str(script),
                          "converter_sha256": sha256(script),
                          "validation": "upstream_converter_completed; not a hardware/runtime validation",
                          "outputs": {name: sha256(temp / name) for name in names}}
            (temp / "conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
            for name in [*names, "conversion.json"]:
                dest = out_dir / name
                # Exclusive publication: never overwrite a concurrent result.
                os.link(temp / name, dest)
                published.append(dest)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        lock.unlink()


def main(argv: list[str] | None = None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    validate_args(args, cli)
    # paths is stdlib-only; do not import rl.py or registration here.
    from orcs.core.paths import DEPS_ROOT
    version_root = DEPS_ROOT / "checkpoints" / "sonic" / args.version
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else version_root / "ported")
    try:
        # Validate conversion support and availability BEFORE acquiring a checkpoint.
        script = find_converter() if args.convert else None
        if script:
            preflight_output(out_dir, args.smpl)
        checkpoint, receipt = resolve_checkpoint(args)
        receipt_file = save_source_receipt(version_root / "sources", receipt)
        print(f"[SONIC] version={args.version}; source={receipt['source_kind']}")
        print(f"[SONIC] checkpoint={checkpoint}")
        print(f"[SONIC] revision={receipt['resolved_revision'] or 'local file; version is user-declared, not verified'}")
        print(f"[SONIC] source receipt={receipt_file}")
        if script:
            convert_checkpoint(script, checkpoint, out_dir, smpl=args.smpl, receipt=receipt)
            print(f"[SONIC] converted outputs={out_dir}")
        else:
            print("[SONIC] weights located; runtime/training compatibility has not been verified")
    except subprocess.CalledProcessError as exc:
        print(f"[SONIC ERROR] converter exited with status {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode if exc.returncode > 0 else 1) from exc
    except Exception as exc:
        print(f"[SONIC ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
