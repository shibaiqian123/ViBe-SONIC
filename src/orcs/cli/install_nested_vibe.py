"""Bind nested ViBe to the outer environment without fetching dependency repos."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def run(*args):
    subprocess.run(args, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orcs-root", type=Path, required=True)
    ap.add_argument("--vibe-root", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    outer = a.orcs_root.resolve()
    inner = (a.vibe_root or outer/"vibe").resolve()
    env_python = outer/".venv/bin/python"
    if Path(sys.prefix).resolve() != (outer/".venv").resolve():
        raise SystemExit(f"run with {env_python}")
    uv = shutil.which("uv") or str(Path.home()/".local/bin/uv")
    if not Path(uv).is_file():
        raise SystemExit("uv executable not found")
    baseline = "69a406a0249ce1da9ac7eba71a33813302aa1ed6"
    run("git", "-C", str(inner), "merge-base", "--is-ancestor", baseline, "HEAD")
    print(f"[binding] ViBe baseline: {baseline}; local changes are retained")
    lock = json.loads((outer/"deps.lock").read_text())
    visual_lock = json.loads((inner/"deps.lock").read_text())
    roots = {"orcs": outer, "vibe": inner}
    for name in ("assets", "mocke", "rsl_rl"):
        if lock[name]["sha"] != visual_lock[name]["sha"]:
            raise SystemExit(f"dependency pins disagree: {name}")
        roots[name] = outer/lock[name]["path"]
        actual = subprocess.check_output(["git", "-C", str(roots[name]), "rev-parse", "HEAD"], text=True).strip()
        if actual != lock[name]["sha"]:
            raise SystemExit(f"dependency revision mismatch: {name}")
    if not a.check:
        dependencies = tomllib.loads((inner/"pyproject.toml").read_text())["project"]["dependencies"]
        try:
            run(uv, "pip", "install", "--python", str(env_python), *dependencies)
        finally:
            # Restore local bindings even if dependency resolution/install fails.
            for name in ("assets", "mocke", "rsl_rl", "orcs", "vibe"):
                run(uv, "pip", "install", "--python", str(env_python), "--no-deps", "-e", str(roots[name]))
    for name, root in roots.items():
        dist = tomllib.loads((root/"pyproject.toml").read_text())["project"]["name"]
        direct = json.loads(importlib.metadata.distribution(dist).read_text("direct_url.json"))
        if direct["url"] != root.as_uri() or not direct.get("dir_info", {}).get("editable"):
            raise SystemExit(f"wrong installed source for {name}: {direct}")
        print(f"[binding] {name}: {root}")
    print("[binding] pinned rsl_rl fork intentionally overrides mjlab PyPI requirement")


if __name__ == "__main__":
    main()
