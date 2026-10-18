"""Read-only GRAIL source audit; output is always a separate diagnostic directory."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

OFFICIAL_COMMIT = "da382dcac447261456314c3e3ce15401d5dc1eb0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_official(root):
    commit = git(root, "rev-parse", "HEAD")
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"official checkout must be at {OFFICIAL_COMMIT}; got {commit}")
    names = ("grail/retargeting/retarget.py", "grail/models/g1_smplx_model.py",
             "grail/visualization/batch_render_replay.py",
             "grail/data_export/export_successful_rollouts.py",
             "data/g1_smplx/g1_smplx_tpose.obj", "data/g1_smplx/g1_smplx_param.npz")
    for name in names:
        committed = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{name}"])
        actual = (root / name).read_bytes()
        if committed.startswith(b"version https://git-lfs.github.com/spec/v1"):
            expected = next(line.split(b":", 1)[1].decode() for line in committed.splitlines() if line.startswith(b"oid sha256:"))
            matches = hashlib.sha256(actual).hexdigest() == expected
        else:
            matches = actual == committed
        if not matches:
            raise ValueError(f"modified or missing LFS official payload: {name}")
    return commit


def unwrap(value, required):
    if not isinstance(value, dict):
        raise ValueError("expected a dictionary")
    if required in value:
        return value
    if len(value) == 1:
        return unwrap(next(iter(value.values())), required)
    raise ValueError(f"ambiguous record: {list(value)}")


def timeline_report(robot, human):
    """Matching endpoints/lengths are not evidence of source-frame correspondence."""
    return {
        "status": "unverified",
        "reason": "Release robot rollout has no verified recon-frame correspondence; endpoint phase is not evidence.",
        "robot_frames": len(robot["dof"]), "robot_fps": float(robot["fps"]),
        "human_frames": len(human["poses"]),
        "human_fps": float(human["mocap_frame_rate"]),
        "robot_duration_s": (len(robot["dof"]) - 1) / float(robot["fps"]),
        "human_duration_s": (len(human["poses"]) - 1) / float(human["mocap_frame_rate"]),
        "requested_staged_frames": [0, 228, 286, 436],
        "corresponding_human_frames": None,
    }


def compare_robot(robot, staged):
    import numpy as np
    from orcs.tasks.perloco.sources.grail import _JOINT_NAMES
    from scipy.spatial.transform import Rotation, Slerp

    with np.load(staged / "motion.npz", allow_pickle=False) as data:
        times = np.arange(len(data["joint_pos"])) / float(data["fps"].reshape(-1)[0])
        source_times = np.arange(len(robot["dof"])) / float(robot["fps"])
        if times[-1] > source_times[-1] + 1e-5:
            raise ValueError("staged robot exceeds source time range")
        times = np.clip(times, source_times[0], source_times[-1])
        names = list(data["joint_names"])
        perm = [list(_JOINT_NAMES).index(n) for n in names]
        expected = np.stack([np.interp(times, source_times, np.asarray(robot["dof"])[:, j]) for j in perm], axis=-1)
        joint_error = np.abs(expected - data["joint_pos"])
        pelvis = list(data["body_names"]).index("pelvis")
        pos = np.stack([np.interp(times, source_times, np.asarray(robot["root_trans_offset"])[:, j]) for j in range(3)], axis=-1)
        pos_error = np.linalg.norm(pos - data["body_pos_w"][:, pelvis], axis=-1)
        expected_rot = Slerp(source_times, Rotation.from_quat(robot["root_rot"]))(times)
        actual_rot = Rotation.from_quat(data["body_quat_w"][:, pelvis][:, [1, 2, 3, 0]])
        angle = (expected_rot.inv() * actual_rot).magnitude()
        return {"root_max_m": float(pos_error.max()), "root_max_deg": float(np.rad2deg(angle.max())),
                "joint_max_deg": float(np.rad2deg(joint_error.max())),
                "pass": bool(pos_error.max() <= .001 and np.rad2deg(angle.max()) <= .1 and np.rad2deg(joint_error.max()) <= .1),
                "scope": "raw release to staged robot; named ORCS joint-order map, not independent robot FK"}


def compare_terrain(path, obj):
    """Reproduce pinned renderer's root T/R/S replacement in an anonymous USD layer.

    Camera centering and common scene yaw are omitted from BOTH sides: those
    rigid transforms cannot change relative geometry or Euclidean errors.
    """
    import numpy as np
    from orcs.tasks.perloco.sources.stair_geometry import (
        static_object_transform,
        usd_meshes,
    )
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("official replay geometry audit currently requires Z-up metre USD")
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("USD has no default prim")
    xf = UsdGeom.Xformable(root)
    ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
    translate = ops.get(UsdGeom.XformOp.TypeTranslate)
    orient = ops.get(UsdGeom.XformOp.TypeOrient)
    scale_op = ops.get(UsdGeom.XformOp.TypeScale)
    if translate is None:
        translate = xf.AddTranslateOp()
    if orient is None:
        orient = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    if scale_op is None:
        scale_op = xf.AddScaleOp()
    positions = np.asarray(obj["root_pos"]).reshape(-1, 3)
    quats = np.asarray(obj["root_quat"]).reshape(-1, 4)
    scale = np.asarray(obj.get("scale", [1., 1., 1.])).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    trans, rot, staged_scale = static_object_transform(obj)
    orcs_vertices = np.concatenate([(v * staged_scale) @ rot.T + trans for v, _ in usd_meshes(path)])
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    errors = []
    first = None
    for position, quat in zip(positions, quats, strict=True):
        translate.Set(Gf.Vec3d(*map(float, position)))
        qt = Gf.Quatf if orient.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd
        orient.Set(qt(*map(float, quat)))
        scale_op.Set(Gf.Vec3f(*map(float, scale)))
        cache = UsdGeom.XformCache()
        points = np.concatenate([np.asarray([cache.GetLocalToWorldTransform(p).Transform(Gf.Vec3d(*map(float, v)))
                                            for v in UsdGeom.Mesh(p).GetPointsAttr().Get()]) for p in meshes])
        if first is None:
            first = points
        if points.shape != orcs_vertices.shape:
            raise ValueError("USD point ordering/shape mismatch")
        errors.append(float(np.linalg.norm(points - orcs_vertices, axis=1).max()))
    return {"first_frame_max_m": errors[0], "all_frames_max_m": max(errors),
            "within_1mm": max(errors) <= .001,
            "scope": "official renderer root TRS replacement vs ORCS static USD vertices; collision decomposition not certified"}, first


def official_forward(root, human, models):
    """Execute the pinned upstream FK function, not the ORCS reimplementation.

    Extracting its AST avoids importing the unrelated IK/viewer/Isaac stack.
    Both standard and G1-template branches are reported; release provenance
    does not establish which generated this particular reconstruction.
    """
    import numpy as np
    import smplx
    import torch

    source = root / "grail/retargeting/retarget.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward_smplx")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    data = {k: torch.as_tensor(np.asarray(human[k]), dtype=torch.float32)
            for k in ("poses", "betas", "trans", "left_hand_pose", "right_hand_pose")}
    model = smplx.create(str(models), model_type="smplx", num_betas=10, num_pca_comps=45)
    standard = ns["forward_smplx"](model, data).joints.numpy()
    trans = np.asarray(human["trans"])[:, None]
    standard = (standard - trans) * np.asarray(human.get("scale", 1.0)) + trans
    spec = importlib.util.spec_from_file_location("grail_audit_g1", root / "grail/models/g1_smplx_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g1 = module.G1ProportionSMPLX.from_g1_smplx_dir(str(models), str(root / "data/g1_smplx"))
    proportioned = ns["forward_smplx"](g1.model, data).joints.numpy()
    return standard, proportioned


def audit(args):
    import joblib
    import numpy as np

    root, output, official = args.source_root.resolve(), args.out_dir.resolve(), args.official_root.resolve()
    if output == root or root in output.parents or (args.staged_sample and
            (output == args.staged_sample.resolve() or args.staged_sample.resolve() in output.parents)):
        raise ValueError("output must be outside source and staged sample directories")
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    commit = verify_official(official)
    files = {k: root / k / (args.sample + (".usd" if k == "object_usd" else ".pkl"))
             for k in ("robot", "objects", "recon", "meta", "object_usd")}
    for path in files.values():
        if not path.is_file() or path.read_bytes()[:43].startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"missing payload: {path}")
    raw = {k: joblib.load(p) for k, p in files.items() if k != "object_usd"}
    robot = unwrap(raw["robot"], "dof")
    human = unwrap(raw["recon"]["human_data"], "poses")
    obj = unwrap(raw["objects"], "root_pos")
    report = {
        "sample": args.sample, "official_commit": commit,
        "gmr_dependency": git(official, "ls-tree", "HEAD", "imports/GMR"),
        "dataset_revision": git(root, "rev-parse", "HEAD"),
        "release_code_provenance": "not established; checkout is a comparison baseline",
        "inputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items()},
        "timeline": timeline_report(robot, human),
        "robot_smpl_placeholder": bool(np.all(np.asarray(robot.get("smpl_joints", [1])) == 0)),
        "scene_scale": float(raw["meta"]["scene_scale"]),
        "human_scale": float(np.asarray(human.get("scale", 1)).reshape(-1)[0]),
        "conventions": {"robot_root_rot": "xyzw (official public release)",
                        "robot_dof": "MuJoCo order (official exporter)",
                        "human_branch": "unknown: compare standard and G1 template without choosing"},
        "smpl_training_ready": False,
        "blockers": ["No verified recon-to-rollout time mapping", "Human template provenance not recorded in release metadata"],
        "checks": {},
    }
    report["official_files"] = {name: sha(official / name) for name in (
        "grail/retargeting/retarget.py", "grail/models/g1_smplx_model.py",
        "grail/visualization/batch_render_replay.py",
        "grail/data_export/export_successful_rollouts.py")}
    if args.staged_sample:
        report["staged_sample"] = str(args.staged_sample.resolve())
        try:
            report["checks"]["robot_conversion"] = compare_robot(robot, args.staged_sample)
        except Exception as exc:
            report["blockers"].append(f"robot comparison: {type(exc).__name__}: {exc}")
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    arrays = {"robot_pos": np.asarray(robot["root_trans_offset"]),
              "robot_quat_wxyz": np.asarray(robot["root_rot"])[:, [3, 0, 1, 2]],
              "robot_dof": np.asarray(robot["dof"]),
              "object_pos": np.asarray(obj["root_pos"]),
              "object_quat": np.asarray(obj["root_quat"])}
    try:
        terrain, points = compare_terrain(files["object_usd"], obj)
        report["checks"]["terrain"] = terrain
        arrays["official_terrain_vertices"] = points
    except Exception as exc:
        report["blockers"].append(f"terrain: {type(exc).__name__}: {exc}")
        report["checks"]["terrain"] = "unverified"
    try:
        standard, g1 = official_forward(official, human, args.body_models)
        from orcs.tasks.perloco.sources.smplx_fk import smpl_channels
        indices = list(range(22)) + [25, 40]
        orcs = smpl_channels(files["recon"], args.body_models)[2]
        arrays.update(official_standard=standard[:, indices], official_g1=g1[:, indices], orcs_human=orcs)
        for label, reference in (("standard", standard), ("g1", g1)):
            errors = np.linalg.norm(reference[:, indices] - orcs, axis=-1)
            report["checks"][f"human_{label}"] = {"max_m": float(errors.max()), "mean_m": float(errors.mean()),
                                                     "within_1mm": bool(errors.max() <= .001)}
        report["checks"]["human_scope"] = "Unpadded source frames; standard applies source scale, G1 branch does not. No terrain or height correction."
    except Exception as exc:
        report["blockers"].append(f"human FK: {type(exc).__name__}: {exc}")
    np.savez_compressed(output / "replay.npz", **arrays)
    report["checks"]["dynamic_contact"] = "not tested: geometric replay only"
    report["status"] = "blocked"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 2


def view(directory, port):
    """Independent source frame sliders deliberately do not imply synchronization."""
    import time

    import joblib
    import mujoco
    import numpy as np
    import viser
    from mjlab.entity import Entity
    from mjviser import ViserMujocoScene
    from orcs.assets import get_g1_flat_hand_cfg
    from orcs.tasks.perloco.sources.grail import _JOINT_NAMES
    from orcs.tasks.perloco.sources.stair_geometry import (
        static_object_transform,
        usd_meshes,
    )
    data = np.load(directory / "replay.npz", allow_pickle=False)
    report = json.loads((directory / "report.json").read_text())
    server = viser.ViserServer(port=port)
    server.gui.add_markdown("**Geometric audit only — no SONIC, no forces.**\n\n"
                            "Human and robot clocks are NOT aligned. Separate frame controls are intentional.")
    names = [k for k in ("official_standard", "official_g1", "orcs_human") if k in data]
    if not names:
        raise ValueError("no human FK results available; inspect report.json")
    selected = server.gui.add_dropdown("Human branch", options=names)
    human_frame = server.gui.add_slider("Human source frame", min=0, max=len(data[names[0]])-1, step=1, initial_value=0)
    robot_frame = server.gui.add_slider("Robot source frame", min=0, max=len(data["robot_pos"])-1, step=1, initial_value=0)
    human = server.scene.add_point_cloud("/human", points=data[names[0]][0], colors=(30, 210, 220), point_size=.025)
    model = Entity(get_g1_flat_hand_cfg()).spec.compile()
    mjdata = mujoco.MjData(model)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in _JOINT_NAMES]
    if any(i < 0 for i in joint_ids):
        raise ValueError("robot model missing expected source joints")
    addresses = model.jnt_qposadr[joint_ids]
    objects = joblib.load(report["inputs"]["objects"]["path"])
    trans, rotation, scale = static_object_transform(objects)
    for i, (vertices, faces) in enumerate(usd_meshes(report["inputs"]["object_usd"]["path"])):
        server.scene.add_mesh_simple(f"/terrain/{i}", vertices=(vertices * scale) @ rotation.T + trans,
                                     faces=faces, color=(100, 150, 200))
    if "official_terrain_vertices" in data:
        server.scene.add_point_cloud("/official_terrain_vertices", points=data["official_terrain_vertices"],
                                     colors=(255, 180, 30), point_size=.02)
    server.gui.add_markdown("Terrain uses the ORCS USD interpretation under audit, first-frame object pose. "
                            "This view is not an independent certification of terrain transforms or contact.")
    status = server.gui.add_markdown("")
    while True:
        hi, ri = int(human_frame.value), int(robot_frame.value)
        human.points = data[selected.value][hi]
        mjdata.qpos[:3] = data["robot_pos"][ri]
        mjdata.qpos[3:7] = data["robot_quat_wxyz"][ri]
        mjdata.qpos[addresses] = data["robot_dof"][ri]
        mujoco.mj_forward(model, mjdata)
        scene.update_from_mjdata(mjdata)
        status.content = (f"Human {hi / report['timeline']['human_fps']:.3f}s; "
                          f"robot {ri / report['timeline']['robot_fps']:.3f}s — correspondence unverified")
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, help="view an existing audit directory")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--sample")
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--body-models", type=Path)
    parser.add_argument("--staged-sample", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.view:
        return view(args.view, args.port)
    if not all((args.source_root, args.sample, args.official_root, args.body_models, args.out_dir)):
        parser.error("audit requires --source-root, --sample, --official-root, --body-models, --out-dir")
    if Path(args.sample).name != args.sample:
        parser.error("sample must be a filename stem")
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
