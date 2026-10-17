"""Geometry-based support labels and fixed-foot targets for stair retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

CONTACT_DISTANCE_M = 0.003
NORMAL_ANGLE_DEG = 30.0
TRANSITION_S = 0.1


@dataclass(frozen=True)
class SurfaceSample:
    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if point.shape != (3,) or normal.shape != (3,):
            raise ValueError("surface point/normal must have shape (3,)")
        length = float(np.linalg.norm(normal))
        if not np.isfinite(point).all() or not np.isfinite(normal).all() or length < 1e-8:
            raise ValueError("surface point/normal must be finite and non-zero")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal / length)


def label_support(
    foot_positions_w: np.ndarray,
    foot_velocities_w: np.ndarray,
    surfaces: list[list[SurfaceSample]],
    *,
    contact_distance_m: float = CONTACT_DISTANCE_M,
    normal_angle_deg: float = NORMAL_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label slow, near-horizontal foot contacts against supplied surfaces."""
    pos = np.asarray(foot_positions_w, dtype=float)
    vel = np.asarray(foot_velocities_w, dtype=float)
    if pos.ndim != 3 or pos.shape[1:] != (2, 3) or vel.shape != pos.shape:
        raise ValueError("foot positions/velocities must be (T,2,3)")
    if len(surfaces) != pos.shape[0]:
        raise ValueError("one surface list is required per frame")
    mask = np.zeros(pos.shape[:2], dtype=bool)
    points = np.full_like(pos, np.nan)
    normals = np.full_like(pos, np.nan)
    limit = math.cos(math.radians(normal_angle_deg))
    for frame, frame_surfaces in enumerate(surfaces):
        for foot in range(2):
            candidates = [
                (surface, abs(float(np.dot(pos[frame, foot] - surface.point, surface.normal))))
                for surface in frame_surfaces
            ]
            if not candidates:
                continue
            surface, distance = min(candidates, key=lambda item: item[1])
            if distance > contact_distance_m or surface.normal[2] < limit:
                continue
            if abs(float(np.dot(vel[frame, foot], surface.normal))) > 0.25:
                continue
            mask[frame, foot] = True
            points[frame, foot] = surface.point
            normals[frame, foot] = surface.normal
    return mask, points, normals


def generate_support_targets(
    ankle_positions_w: np.ndarray,
    support_mask: np.ndarray,
    surface_points_w: np.ndarray,
    surface_normals_w: np.ndarray,
    *,
    ankle_to_sole_w: np.ndarray,
    fps: float,
    transition_s: float = TRANSITION_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold one tread target throughout each support run and blend transitions."""
    ankle = np.asarray(ankle_positions_w, dtype=float)
    mask = np.asarray(support_mask, dtype=bool)
    points = np.asarray(surface_points_w, dtype=float)
    normals = np.asarray(surface_normals_w, dtype=float)
    offset = np.asarray(ankle_to_sole_w, dtype=float)
    if ankle.ndim != 3 or ankle.shape[1:] != (2, 3):
        raise ValueError("ankle_positions_w must be (T,2,3)")
    if mask.shape != ankle.shape[:2] or points.shape != ankle.shape or normals.shape != ankle.shape:
        raise ValueError("support arrays have incompatible shapes")
    if offset.shape != (2, 3) or fps <= 0 or transition_s < 0:
        raise ValueError("invalid ankle offset, fps, or transition")
    target = ankle.copy()
    weights = np.zeros(mask.shape, dtype=float)
    transition_frames = max(1, int(round(transition_s * fps)))
    for foot in range(2):
        frame = 0
        while frame < len(mask):
            if not mask[frame, foot] or not np.isfinite(points[frame, foot]).all():
                frame += 1
                continue
            start = frame
            while frame + 1 < len(mask) and mask[frame + 1, foot]:
                frame += 1
            end = frame
            fixed = points[start, foot] - offset[foot]
            for index in range(start, end + 1):
                target[index, foot] = fixed
                enter = min(1.0, (index - start + 1) / transition_frames)
                leave = min(1.0, (end - index + 1) / transition_frames)
                weights[index, foot] = min(enter, leave)
            frame += 1
    return target, weights
