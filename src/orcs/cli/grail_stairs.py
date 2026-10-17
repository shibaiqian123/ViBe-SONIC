"""Resolve and fetch a small, pinned GRAIL stair selection (no simulator imports).

The selection JSON is the receipt shared by fetching, staging and provenance.
Only Git metadata is fetched to select samples; LFS payloads are explicitly scoped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

REPOSITORY = "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
ALIASES = {"stair1": "stair_p1", "stair2": "stair_p2"}
CATEGORIES = ("stair_p1", "stair_p2")
SCHEMA = 1


def categories(value: str) -> tuple[str, ...]:
    result = tuple(sorted({ALIASES.get(x.strip(), x.strip()) for x in value.split(",")}))
    if not result or any(x not in CATEGORIES for x in result):
        raise ValueError(f"expected stair1,stair2 or stair_p1,stair_p2; got {value!r}")
    return result


def safe_path(value: str) -> str:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\n" in value:
        raise ValueError(f"unsafe dataset path: {value!r}")
    return str(p)


def select_samples(files: list[str], cats: tuple[str, ...], roster: dict) -> list[dict]:
    available = set(files)
    explicit = roster.get("samples")
    candidates = []
    for category in cats:
        prefix = f"data/{category}/robot/"
        for name in sorted(available):
            if not name.startswith(prefix) or not name.endswith(".pkl"):
                continue
            stem = PurePosixPath(name).stem
            parts = stem.split("__")
            if len(parts) != 3:
                continue
            candidates.append({"category": category, "stem": stem, "family": parts[1]})
    if explicit is not None:
        wanted = {(x["category"], x["stem"]) for x in explicit}
        candidates = [x for x in candidates if (x["category"], x["stem"]) in wanted]
        if {(x["category"], x["stem"]) for x in candidates} != wanted:
            raise ValueError("roster names missing samples or categories outside --grail-categories")
    complete = []
    for sample in candidates:
        base = f"data/{sample['category']}"
        stem = sample["stem"]
        paths = {d: f"{base}/{d}/{stem}.pkl" for d in ("robot", "recon", "objects", "meta")}
        paths["object_usd"] = f"{base}/object_usd/{stem}.usd"
        if all(p in available for p in paths.values()):
            complete.append({**sample, "files": paths})
        elif explicit is not None:
            raise ValueError(f"incomplete sample {stem}: {[p for p in paths.values() if p not in available]}")
    chosen = []
    for category in cats:
        families = sorted({x["family"] for x in complete if x["category"] == category})
        if explicit is None:
            n_clips = int(roster.get("clips_per_family", 2))
            n_families = int(roster.get("families_per_category", 2))
            if n_clips < 1 or n_families < 1:
                raise ValueError("selection limits must be positive")
            if not roster.get("all_samples", False):
                families = [f for f in families if sum(x["category"] == category and x["family"] == f
                                                      for x in complete) >= n_clips]
                families = families[:n_families]
                if len(families) < n_families:
                    raise ValueError(f"{category}: not enough complete families; use an explicit roster")
        for family in families:
            rows = [x for x in complete if x["category"] == category and x["family"] == family]
            if explicit is None and not roster.get("all_samples", False):
                rows = rows[:int(roster.get("clips_per_family", 2))]
            chosen.extend(rows)
        if not any(x["category"] == category for x in chosen):
            raise ValueError(f"no complete stair samples in {category}")
    if not chosen:
        raise ValueError("selection is empty")
    # Each take owns its geometry: never assume two takes share scale/pose.
    for sample in chosen:
        key = f"{sample['category']}/{sample['stem']}"
        sample["tile_id"] = sample['category'] + "_" + hashlib.sha256(key.encode()).hexdigest()[:16]
        sample["sample_id"] = "sample0"
    return chosen


def payload_paths(selection: dict, smpl: bool, video: bool) -> list[str]:
    paths = set(selection.get("dependencies", []))
    for row in selection["samples"]:
        paths.update(p for kind, p in row["files"].items() if kind != "recon" or smpl)
        if video:
            paths.add(f"data/{row['category']}/video/{row['stem']}.mp4")
    return sorted(safe_path(p) for p in paths)


def check_payload(root: Path, paths: list[str]) -> None:
    for name in paths:
        p = root / safe_path(name)
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty source file: {p}")
        with p.open("rb") as f:
            if f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"unresolved Git LFS pointer: {p}")


def _git(root: Path, *args: str, capture: bool = False, stdin: str | None = None) -> str:
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    p = subprocess.run(["git", "-C", str(root), *args], check=True, env=env,
                       text=True, input=stdin, stdout=subprocess.PIPE if capture else None)
    return p.stdout.strip() if capture else ""


def materialize_lfs(root: Path, name: str) -> None:
    """Smudge one selected pointer; never scan refs in a partial clone."""
    target = root / safe_path(name)
    with target.open("rb") as stream:
        pointer = stream.read(1024)
    if not pointer.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return
    oid = re.search(rb"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
    size = re.search(rb"^size ([0-9]+)$", pointer, re.MULTILINE)
    if oid is None or size is None or target.stat().st_size > 1024:
        raise ValueError(f"invalid Git LFS pointer: {target}")
    expected_size = int(size[1])
    print(f"[LFS] {name} ({expected_size / 1024**2:.2f} MiB)", flush=True)
    env = dict(os.environ)
    # Checkout deliberately skips smudging; this explicit request must not.
    env.pop("GIT_LFS_SKIP_SMUDGE", None)
    env.pop("GIT_LFS_SKIP_DOWNLOAD_ERRORS", None)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".orcs-lfs-", delete=False) as out:
            temporary = Path(out.name)
            subprocess.run(
                ["git", "-C", str(root), "-c", "lfs.skipdownloaderrors=false",
                 "-c", "lfs.fetchinclude=", "-c", "lfs.fetchexclude=",
                 "lfs", "smudge", "--", name],
                input=pointer, stdout=out, env=env, check=True,
            )
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if temporary.stat().st_size != expected_size or digest.hexdigest() != oid[1].decode():
            raise ValueError(f"Git LFS payload size/SHA-256 mismatch: {target}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_files(root: Path, paths: list[str], revision: str) -> None:
    # lfs pull scans ls-tree -l across the ref, causing lazy blob fetches even
    # with --include. Feed only selected pointers to smudge instead: LFS uses
    # their object IDs and reuses its existing cache without enumerating refs.
    paths = [safe_path(name) for name in paths]
    for start in range(0, len(paths), 100):
        batch = paths[start:start + 100]
        _git(root, "--literal-pathspecs", "checkout", revision, "--", *batch)
        for name in batch:
            materialize_lfs(root, name)
    check_payload(root, paths)


def _shading_assets(layer) -> set[str]:
    """Identify asset-valued shader attributes, not arbitrary image-like paths."""
    from pxr import Sdf
    shading, other = set(), set(layer.GetCompositionAssetDependencies())

    def assets(value):
        if isinstance(value, Sdf.AssetPath):
            return {value.path}
        if isinstance(value, (list, tuple)) or type(value).__name__ == "AssetPathArray":
            return {item.path for item in value if isinstance(item, Sdf.AssetPath)}
        return set()

    def visit(path):
        attr = layer.GetObjectAtPath(path)
        if not isinstance(attr, Sdf.AttributeSpec):
            return
        values = assets(attr.default)
        for time in layer.ListTimeSamplesForPath(path):
            values.update(assets(layer.QueryTimeSample(path, time)))
        target = shading if attr.owner.typeName in {"Shader", "Material", "NodeGraph"} else other
        target.update(values)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return shading - other


def usd_dependencies(root: Path, selection: dict) -> list[str]:
    """Collect geometry dependencies; ignore shader-only textures/MDL programs.

    Collision conversion reads mesh geometry, not rendering materials. Published
    USDs can retain author-machine material paths. Never exempt composition
    references or non-shader assets merely because their suffix looks visual.
    """
    from pxr import Sdf, UsdUtils
    result = set()
    shading_by_layer = {}
    ignored = set()
    visual_suffixes = {".mdl", ".jpg", ".jpeg", ".png", ".exr", ".tga", ".tif",
                       ".tiff", ".bmp", ".hdr", ".webp", ".dds", ".ktx"}

    def process(layer, info):
        authored = info.assetPath
        if layer.identifier not in shading_by_layer:
            shading_by_layer[layer.identifier] = _shading_assets(layer)
        shading = shading_by_layer[layer.identifier]
        if authored in shading and Path(authored).suffix.lower() in visual_suffixes:
            ignored.add(authored)
            return UsdUtils.DependencyInfo("")
        # Anchor unresolved paths to their authoring layer, not the root USD.
        for value in (authored, *info.dependencies):
            if not value:
                continue
            candidate = Path(Sdf.ComputeAssetPathRelativeToLayer(layer, value))
            if not candidate.is_absolute():
                candidate = Path(layer.realPath).parent / candidate
            candidate = candidate.resolve()
            try:
                relative = candidate.relative_to(root.resolve()).as_posix()
            except ValueError as exc:
                raise ValueError(f"USD dependency outside source repository: {value} "
                                 f"(layer {layer.realPath})") from exc
            result.add(safe_path(relative))
        return info

    for sample in selection["samples"]:
        usd = root / safe_path(sample["files"]["object_usd"])
        if Sdf.Layer.FindOrOpen(str(usd)) is None:
            raise ValueError(f"cannot open USD source: {usd}")
        result.add(usd.relative_to(root).as_posix())
        UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(usd)), processingFunc=process)
    if ignored:
        print(f"[USD] skipped {len(ignored)} shader-only texture/MDL dependencies "
              "(collision geometry does not use materials)", flush=True)
    return sorted(result)


def prepare(root: Path, receipt: Path, cats: tuple[str, ...], roster: dict,
            *, smpl: bool, video: bool, offline: bool) -> dict:
    signature = hashlib.sha256(json.dumps([cats, roster], sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        selection = json.loads(receipt.read_text())
        if selection.get("schema") != SCHEMA or selection.get("request") != signature:
            raise ValueError(f"selection differs from {receipt}; use a new --selection path")
    else:
        if offline:
            raise FileNotFoundError(f"--skip-clone requires an existing selection: {receipt}")
        if not (root / ".git").exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(root)],
                           check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        ref = str(roster.get("revision", "HEAD"))
        if ref != "HEAD":
            _git(root, "fetch", "--depth", "1", "origin", ref)
            ref = "FETCH_HEAD"
        revision = _git(root, "rev-parse", ref, capture=True)
        files = _git(root, "ls-tree", "-r", "--name-only", revision, capture=True).splitlines()
        selection = {"schema": SCHEMA, "repository": REPOSITORY, "revision": revision,
                     "request": signature, "samples": select_samples(files, cats, roster),
                     "dependencies": []}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(selection, indent=2) + "\n")
    paths = payload_paths(selection, smpl, video)
    if offline:
        check_payload(root, paths)
        check_payload(root, usd_dependencies(root, selection))
        return selection
    if not (root / ".git").exists():
        raise FileNotFoundError(f"selection exists but source checkout missing: {root}; restore pinned checkout first")
    # Only selected pointers/assets enter this dedicated checkout. Payload fetch is scoped too.
    _git(root, "sparse-checkout", "set", "--no-cone", "--stdin", stdin="\n".join('/'+p for p in paths)+"\n")
    _git(root, "checkout", "--detach", selection["revision"])
    fetch_files(root, paths, selection["revision"])
    for _ in range(16):
        dependencies = usd_dependencies(root, selection)
        missing = sorted(set(dependencies) - set(paths))
        if not missing:
            break
        paths = sorted(set(paths) | set(missing))
        _git(root, "sparse-checkout", "set", "--no-cone", "--stdin", stdin="\n".join('/'+p for p in paths)+"\n")
        fetch_files(root, missing, selection["revision"])
        selection["dependencies"] = sorted(set(selection["dependencies"]) | set(missing))
        receipt.write_text(json.dumps(selection, indent=2) + "\n")
    else:
        raise ValueError("USD dependency closure did not converge")
    check_payload(root, paths)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--categories", default="stair1,stair2")
    parser.add_argument("--no-smpl", action="store_true")
    parser.add_argument("--with-video", action="store_true")
    parser.add_argument("--skip-clone", action="store_true")
    args = parser.parse_args()
    roster = tomllib.loads(args.roster.read_text())
    selection = prepare(args.root, args.selection, categories(args.categories), roster,
                        smpl=not args.no_smpl, video=args.with_video, offline=args.skip_clone)
    print(f"[stairs] {len(selection['samples'])} samples at {selection['revision']}; selection {args.selection}")


if __name__ == "__main__":
    main()
