"""Small offline fixtures for the stair ingestion contracts; no release payloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


fetch = module("stair_fetch", "src/orcs/cli/grail_stairs.py")
geometry = module("stair_geometry", "src/orcs/tasks/perloco/sources/stair_geometry.py")
receipt = module("seed_receipt", "src/orcs/core/data/seed_receipt.py")


def source_files(category="stair_p1", family="stairs_1", takes=2):
    return [f"data/{category}/{kind}/terrain_stairs__{family}__{i:04d}.{ext}"
            for i in range(takes)
            for kind, ext in (("robot", "pkl"), ("recon", "pkl"), ("objects", "pkl"),
                              ("meta", "pkl"), ("object_usd", "usd"))]


def test_aliases_and_rejected_mixed_sources():
    assert fetch.categories("stair2,stair1,stair_p1") == ("stair_p1", "stair_p2")
    with pytest.raises(ValueError):
        fetch.categories("curb,stair1")


def test_default_selection_is_sorted_complete_and_namespaced():
    files = []
    for cat in fetch.CATEGORIES:
        for family in ("c", "a", "b"):
            files += source_files(cat, family)
    selected = fetch.select_samples(list(reversed(files)), fetch.CATEGORIES, {})
    assert len(selected) == 8
    assert {x['family'] for x in selected} == {"a", "b"}
    assert len({x['tile_id'] for x in selected}) == 8
    assert selected == fetch.select_samples(files, fetch.CATEGORIES, {})


def test_explicit_incomplete_selection_errors():
    files = source_files()
    files.pop()
    with pytest.raises(ValueError, match="incomplete"):
        fetch.select_samples(files, ("stair_p1",), {"samples": [
            {"category": "stair_p1", "stem": "terrain_stairs__stairs_1__0001"}]})


def test_payload_no_smpl_and_video_are_exact():
    rows = fetch.select_samples(source_files(), ("stair_p1",), {"families_per_category": 1})
    selection = {"samples": rows, "dependencies": ["data/stair_p1/object_usd/textures/a.png"]}
    paths = fetch.payload_paths(selection, False, False)
    assert not any('/recon/' in x or '/video/' in x or '*' in x for x in paths)
    assert len(paths) == 9
    full = fetch.payload_paths(selection, True, True)
    assert len(full) == 13


def test_lfs_pointer_and_missing_file_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch.check_payload(tmp_path, ["a"])
    (tmp_path / 'a').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\n')
    with pytest.raises(ValueError, match='pointer'):
        fetch.check_payload(tmp_path, ["a"])
    (tmp_path / 'a').write_bytes(b'payload')
    fetch.check_payload(tmp_path, ["a"])


def test_no_payload_escape():
    for p in ('/etc/passwd', '../data/a', 'a/../../b'):
        with pytest.raises(ValueError):
            fetch.safe_path(p)


def test_static_pose_and_dynamic_rejection():
    t, r, s = geometry.static_object_transform({"obj_data": {
        "obj_R": np.tile(np.eye(3), (3, 1, 1)), "obj_t": np.tile([1, 2, 3], (3, 1)),
        "obj_scale": 2.0}})
    np.testing.assert_allclose(t, [1, 2, 3])
    np.testing.assert_allclose(r, np.eye(3))
    np.testing.assert_allclose(s, [2, 2, 2])
    with pytest.raises(ValueError, match='moving'):
        geometry.static_object_transform({'object_pos_w': [[0, 0, 0], [0, 0, .1]],
                                          'object_quat_w': [[1, 0, 0, 0]] * 2})
    with pytest.raises(ValueError, match='quat_order'):
        geometry.static_object_transform({'obj_pos': [0, 0, 0], 'obj_rot': [0, 0, 0, 1]})


def test_seed_receipt_detects_changed_source(tmp_path):
    sample = tmp_path / 'sample0'
    sample.mkdir()
    (tmp_path / 'tile.json').write_text('{}')
    (sample / 'metadata.json').write_text('{}')
    (sample / 'smpl_motion.npz').write_bytes(b'fixture')
    assert not receipt.seed_matches_source(sample)
    receipt.write_seed_receipt(sample)
    assert receipt.seed_matches_source(sample)
    (tmp_path / 'tile.json').write_text('{"changed":true}')
    assert not receipt.seed_matches_source(sample)


def test_usd_hierarchy_units_and_up_axis(tmp_path):
    from pxr import Gf, Usd, UsdGeom
    p = tmp_path / 'tiny.usda'
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, 'Y')
    parent = UsdGeom.Xform.Define(stage, '/Root')
    parent.AddTranslateOp().Set(Gf.Vec3d(100, 200, 300))
    mesh = UsdGeom.Mesh.Define(stage, '/Root/Mesh')
    mesh.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    v, f = geometry.usd_meshes(p)[0]
    np.testing.assert_allclose(v, [[1, -3, 2], [2, -3, 2], [1, -3, 3]])
    assert f.tolist() == [[0, 1, 2]]


def test_convex_box_and_mujoco_mesh_compile():
    import mujoco
    import trimesh
    box = trimesh.creation.box(extents=[1, 1, .2])
    parts = geometry.collision_parts(box.vertices, box.faces)
    assert len(parts) == 1
    spec = mujoco.MjSpec()
    v, f = parts[0]
    spec.add_mesh(name='piece', uservert=v.ravel().tolist(), userface=f.ravel().tolist())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='piece')
    model = spec.compile()
    assert model.ngeom == 1


def test_concave_steps_not_one_hull():
    import trimesh
    # Extrude an L-shaped, two-step cross section (explicit boundary triangles).
    polygon = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    v = np.array([[x, y, z] for y in (-.5, .5) for x, z in polygon])
    faces = []
    cap = [(0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5)]
    faces += cap
    faces += [tuple(i+6 for i in t[::-1]) for t in cap]
    for i in range(6):
        j = (i+1) % 6
        faces += [(i, i+6, j+6), (i, j+6, j)]
    mesh = trimesh.Trimesh(v, faces, process=True)
    mesh.fix_normals()
    parts = geometry.collision_parts(mesh.vertices, mesh.faces)
    assert len(parts) >= 2
    volume = sum(abs(trimesh.Trimesh(a, b).volume) for a, b in parts)
    assert abs(volume - abs(mesh.volume)) / abs(mesh.volume) < .1


def test_pinned_request_cannot_silently_change(tmp_path):
    p = tmp_path / 'selection.json'
    p.write_text(json.dumps({'schema': 1, 'request': 'different'}))
    with pytest.raises(ValueError, match='selection differs'):
        fetch.prepare(tmp_path, p, ('stair_p1',), {}, smpl=False, video=False, offline=True)


def test_grail_released_object_pose_schema():
    data = {"take": {"root_pos": np.array([[[1., 2., 3.]]] * 4),
                     "root_quat": np.array([[[1., 0., 0., 0.]]] * 4),
                     "scale": [2., 3., 4.], "fps": 25}}
    t, r, scale = geometry.static_object_transform(data)
    np.testing.assert_allclose(t, [1, 2, 3])
    np.testing.assert_allclose(r, np.eye(3))
    np.testing.assert_allclose(scale, [2, 3, 4])


def test_runtime_builds_separate_step_collision_surfaces(tmp_path, monkeypatch):
    """Exercise real MuJoCo compilation/rays, stubbing only MJLab config containers."""
    import sys
    import types
    from dataclasses import dataclass

    import mujoco
    import trimesh

    @dataclass(kw_only=True)
    class BaseCfg:
        size: tuple = (6., 6.)

    @dataclass
    class Geometry:
        geom: object
        color: object

    @dataclass
    class Output:
        origin: object
        geometries: object

    fake = types.ModuleType('mjlab.terrains.terrain_generator')
    fake.SubTerrainCfg = BaseCfg
    fake.TerrainGeneratorCfg = BaseCfg
    fake.TerrainGeometry = Geometry
    fake.TerrainOutput = Output
    monkeypatch.setitem(sys.modules, 'mjlab.terrains.terrain_generator', fake)
    roster = types.ModuleType('orcs.tasks.perloco.roster')
    roster.Roster = object
    monkeypatch.setitem(sys.modules, 'orcs.tasks.perloco.roster', roster)
    name = 'standalone_stair_terrain'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src/orcs/tasks/perloco/terrain.py')
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, runtime)
    spec.loader.exec_module(runtime)
    tile_dir = tmp_path / 'test' / 'level_0.00'
    tile_dir.mkdir(parents=True)
    pieces = []
    for x, height in ((-.5, .2), (.5, .4)):
        mesh = trimesh.creation.box(extents=[1., 1., height])
        mesh.apply_translation([x, 0., height / 2])
        pieces.append({'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist()})
    (tile_dir / 'tile.json').write_text(json.dumps({'boxes': [], 'meshes': pieces, 'size': [8., 6.]}))
    assert runtime.staged_tile_size(tmp_path) == (8., 6.)
    scene = mujoco.MjSpec()
    scene.worldbody.add_body(name='terrain')
    result = runtime.TileTerrainCfg(root=tmp_path, family='test', levels=(0.,), size=(8., 6.)).function(
        0., scene, np.random.default_rng(0))
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for x, expected in ((3.5, .2), (4.5, .4)):
        distance = mujoco.mj_ray(model, data, np.array([x, 3., 2.]), np.array([0., 0., -1.]),
                                 None, 1, -1, np.zeros(1, dtype=np.int32))
        assert 2. - distance == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(result.origin, [4., 3., 0.])


def test_usd_relative_texture_dependency_is_scoped(tmp_path):
    from pxr import Sdf, Usd
    folder = tmp_path / 'data/stair_p1/object_usd'
    folder.mkdir(parents=True)
    path = folder / 'sample.usda'
    stage = Usd.Stage.CreateNew(str(path))
    prim = stage.DefinePrim('/Material', 'Shader')
    prim.CreateAttribute('inputs:file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('textures/a.png'))
    stage.GetRootLayer().Save()
    selection = {'samples': [{'files': {'object_usd': path.relative_to(tmp_path).as_posix()}}]}
    deps = fetch.usd_dependencies(tmp_path, selection)
    assert 'data/stair_p1/object_usd/textures/a.png' in deps
    assert all(not Path(p).is_absolute() for p in deps)
