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
    from scipy.spatial.transform import Rotation, Slerp

    from orcs.tasks.perloco.sources.grail import _JOINT_NAMES

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
    from pxr import Gf, Usd, UsdGeom

    from orcs.tasks.perloco.sources.stair_geometry import (
        static_object_transform,
        usd_meshes,
    )

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


def official_forward(root, human, models, mesh_output=None):
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
    standard_output = ns["forward_smplx"](model, data, return_mesh=mesh_output is not None)
    standard = standard_output.joints.numpy()
    trans = np.asarray(human["trans"])[:, None]
    standard = (standard - trans) * np.asarray(human.get("scale", 1.0)) + trans
    spec = importlib.util.spec_from_file_location("grail_audit_g1", root / "grail/models/g1_smplx_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g1 = module.G1ProportionSMPLX.from_g1_smplx_dir(str(models), str(root / "data/g1_smplx"))
    g1_output = ns["forward_smplx"](g1.model, data, return_mesh=mesh_output is not None)
    proportioned = g1_output.joints.numpy()
    if mesh_output is not None:
        mesh_output["official_standard_vertices"] = ((standard_output.vertices.numpy() - trans)
                                                     * np.asarray(human.get("scale", 1.0)) + trans).astype(np.float32)
        mesh_output["official_g1_vertices"] = g1_output.vertices.numpy().astype(np.float32)
        mesh_output["human_faces"] = np.asarray(model.faces, dtype=np.int32)
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
        standard, g1 = official_forward(official, human, args.body_models, arrays)
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
    write_foot_diagnostics(output, arrays, report)
    np.savez_compressed(output / "replay.npz", **arrays)
    report["checks"]["dynamic_contact"] = "not tested: geometric replay only"
    report["status"] = "blocked"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 2


def foot_diagnostics(joints, fps):
    """Anatomical joint metrics, NOT sole contact or support-phase labels."""
    import numpy as np
    if fps <= 0 or len(joints) < 2:
        raise ValueError("foot diagnostics require positive fps and at least two frames")
    ankle = joints[:, [7, 8]]
    foot = joints[:, [10, 11]]
    delta = foot - ankle
    angle = np.degrees(np.arctan2(delta[..., 2], np.linalg.norm(delta[..., :2], axis=-1)))
    speed = np.linalg.norm(np.gradient(foot, 1 / fps, axis=0), axis=-1)
    return ankle, foot, angle, speed


def write_foot_diagnostics(output, arrays, report):
    import csv

    import numpy as np
    fps = report["timeline"]["human_fps"]
    summary = {"scope": "Joint-axis diagnostics only; ankle is not heel, foot joint is not toe/sole. "
               "No verified support labels: cannot classify persistent tiptoe vs push-off.", "branches": {}}
    for branch in ("official_standard", "official_g1"):
        if branch not in arrays:
            continue
        ankle, foot, angle, speed = foot_diagnostics(arrays[branch], fps)
        with (output / f"{branch}_feet.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "time_s", "side", "ankle_z_m", "foot_joint_z_m",
                             "ankle_to_foot_axis_deg", "foot_joint_speed_m_s"])
            for frame in range(len(ankle)):
                for side, label in enumerate(("left", "right")):
                    writer.writerow([frame, frame / fps, label, ankle[frame, side, 2],
                                     foot[frame, side, 2], angle[frame, side], speed[frame, side]])
        summary["branches"][branch] = {
            "frames": len(ankle), "axis_angle_range_deg": [float(np.min(angle)), float(np.max(angle))],
            "support_phase": "unverified", "tiptoe_classification": "not established"}
    (output / "foot_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")


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
    with np.load(directory / "replay.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
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
    play = server.gui.add_checkbox("Play human (independent clock)", initial_value=False)
    show_mesh = server.gui.add_checkbox("Show human mesh", initial_value=True)
    show_labels = server.gui.add_checkbox("Show ankle / foot labels", initial_value=True)
    mesh = None
    if "human_faces" in data and "official_standard_vertices" in data:
        mesh = server.scene.add_mesh_simple("/human_mesh", vertices=data["official_standard_vertices"][0],
                                            faces=data["human_faces"], color=(150, 200, 210), opacity=.65)
    else:
        server.gui.add_markdown("**Old audit: mesh unavailable. Re-run audit into a NEW output directory.**")
    labels = [server.scene.add_label(f"/foot_labels/{i}", text=text)
              for i, text in ((7, "L ankle (7)"), (8, "R ankle (8)"),
                              (10, "L foot joint (10)"), (11, "R foot joint (11)"))]
    server.gui.add_markdown("Labels are anatomical joints, NOT heel/toe contact points. "
                            "Mesh shows the actual reconstructed foot surface. No support phase is verified.")
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
    previous = None
    tick = time.monotonic()
    while True:
        now = time.monotonic()
        if play.value:
            steps = int((now - tick) * report["timeline"]["human_fps"])
            if steps:
                human_frame.value = (int(human_frame.value) + steps) % len(data[selected.value])
                tick += steps / report["timeline"]["human_fps"]
        else:
            tick = now
        hi, ri = int(human_frame.value), int(robot_frame.value)
        current = (hi, ri, selected.value, show_mesh.value, show_labels.value)
        if current == previous:
            time.sleep(.02)
            continue
        previous = current
        human.points = data[selected.value][hi]
        for label, index in zip(labels, (7, 8, 10, 11), strict=True):
            label.position = data[selected.value][hi, index]
            label.visible = show_labels.value
        if mesh is not None:
            mesh_key = selected.value + "_vertices"
            mesh.visible = show_mesh.value and mesh_key in data
            if mesh_key in data:
                mesh.vertices = data[mesh_key][hi]
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
