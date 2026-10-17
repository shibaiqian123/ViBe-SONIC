"""USD stair geometry -> world-metre triangles -> convex collision pieces.

No simulator or task discovery imports. Unknown/dynamic transforms fail closed.
"""
from __future__ import annotations

import numpy as np


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("expected finite wxyz quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3, 3))


def static_object_transform(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known explicit schemas only; never guess quaternion order from values."""
    if "obj_data" in data:
        data = data["obj_data"]
    elif len(data) == 1 and isinstance(next(iter(data.values())), dict):
        return static_object_transform(next(iter(data.values())))
    if "obj_R" in data and "obj_t" in data:
        rot, pos = np.asarray(data["obj_R"]), np.asarray(data["obj_t"])
    elif "root_pos" in data and "root_quat" in data:
        # GRAIL export_successful_rollouts.trajectory_to_object_pkl:
        # (T,1,3), (T,1,4), wxyz, with object-local scale.
        pos = np.asarray(data["root_pos"])
        rot = quaternion_matrix(data["root_quat"])
    elif "object_pos_w" in data and "object_quat_w" in data:
        pos = np.asarray(data["object_pos_w"])
        rot = quaternion_matrix(data["object_quat_w"])
    elif "obj_pos_w" in data and "obj_quat_w" in data:
        pos = np.asarray(data["obj_pos_w"])
        rot = quaternion_matrix(data["obj_quat_w"])
    elif "obj_pos" in data and "obj_rot" in data:
        pos = np.asarray(data["obj_pos"])
        order = data.get("quat_order")
        if order not in ("wxyz", "xyzw"):
            raise ValueError("obj_rot needs an explicit quat_order (wxyz or xyzw)")
        q = np.asarray(data["obj_rot"])
        rot = quaternion_matrix(q if order == "wxyz" else q[..., [3, 0, 1, 2]])
    else:
        raise ValueError(f"unsupported stair object pose schema: {sorted(data)}")
    pos, rot = pos.reshape(-1, 3), rot.reshape(-1, 3, 3)
    scale = np.asarray(data.get("obj_scale", data.get("object_scale", data.get("scale", 1.0))), dtype=float)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    scale = scale.reshape(-1, 3)
    if not all(np.isfinite(x).all() for x in (pos, rot, scale)) or np.any(scale <= 0):
        raise ValueError("invalid stair pose/scale")
    if not (np.allclose(pos, pos[0], atol=1e-3) and np.allclose(rot, rot[0], atol=1e-3)
            and np.allclose(scale, scale[0], atol=1e-6)):
        raise ValueError("moving stair asset cannot be staged as static terrain")
    if not np.allclose(rot[0].T @ rot[0], np.eye(3), atol=1e-4) or np.linalg.det(rot[0]) < 0.99:
        raise ValueError("invalid object rotation matrix")
    return pos[0], rot[0], scale[0]


def usd_meshes(path):
    """Bake authored local-to-world transforms, metersPerUnit and upAxis once."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open USD {path}")
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    up = str(UsdGeom.GetStageUpAxis(stage)).upper()
    if up not in ("Y", "Z") or not np.isfinite(unit) or unit <= 0:
        raise ValueError(f"unsupported USD convention: {up}, {unit}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):
            raise ValueError("point-instanced stair USD is unsupported; flatten instances first")
        if prim.IsInstance():
            raise ValueError("instanced stair USD is unsupported; flatten instances first")
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetPointsAttr().GetNumTimeSamples() > 1:
            raise ValueError("animated terrain mesh is unsupported")
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if UsdGeom.Xformable(ancestor).TransformMightBeTimeVarying():
                raise ValueError("animated terrain transform is unsupported")
            ancestor = ancestor.GetParent()
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:, :3] * unit
        if up == "Y":
            vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        triangles, offset = [], 0
        for n in counts:
            face = indices[offset:offset+n]
            if n < 3:
                raise ValueError("degenerate USD face")
            # Triangles and convex quads only; arbitrary n-gons need triangulation upstream.
            if n > 4:
                raise ValueError("USD faces with >4 vertices must be triangulated upstream")
            triangles.extend((face[0], face[j], face[j+1]) for j in range(1, n-1))
            offset += n
        faces = np.asarray(triangles, dtype=np.int32)
        if str(mesh.GetOrientationAttr().Get()) == "leftHanded":
            faces = faces[:, ::-1]
        if np.linalg.det(transform[:3, :3]) < 0:
            faces = faces[:, ::-1]
        if not len(faces) or not np.isfinite(vertices).all() or indices.min() < 0 or indices.max() >= len(vertices):
            raise ValueError("invalid USD mesh")
        meshes.append((vertices, faces))
    if not meshes:
        raise ValueError(f"USD has no meshes: {path}")
    return meshes


def collision_parts(vertices, faces):
    """Keep convex connected components exact; decompose concave ones with CoACD."""
    import coacd
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    parts = []
    for component in mesh.split(only_watertight=False):
        if not component.is_watertight or component.volume <= 1e-10:
            raise ValueError("stair collision surface must be watertight with positive volume")
        if component.is_convex:
            parts.append((np.asarray(component.vertices), np.asarray(component.faces)))
        else:
            output = coacd.run_coacd(coacd.Mesh(np.asarray(component.vertices), np.asarray(component.faces)),
                                     threshold=0.01, seed=0)
            if len(output) < 2:
                raise ValueError("concave stair decomposed to one hull; refusing to erase steps")
            parts.extend(output)
    if not parts:
        raise ValueError("no convex collision pieces")
    for v, f in parts:
        hull = trimesh.Trimesh(vertices=v, faces=f, process=True)
        if not hull.is_convex or not hull.is_watertight:
            raise ValueError("invalid convex decomposition output")
    return tuple((np.asarray(v, dtype=float), np.asarray(f, dtype=np.int32)) for v, f in parts)
