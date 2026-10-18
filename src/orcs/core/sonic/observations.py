"""SONIC reference computation, independent of MJLab for numerical verification."""
from __future__ import annotations

import torch

from .specs import WRIST_MJ, get_spec


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_inv(q):
    return q * q.new_tensor([1, -1, -1, -1]) / q.square().sum(-1, keepdim=True)


def matrix(q):
    # Same normalized quaternion rotation convention as IsaacLab.
    w, x, y, z = q.unbind(-1)
    s = 2.0 / q.square().sum(-1)
    return torch.stack((1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
                        s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
                        s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def heading(q):
    forward = matrix(q)[..., :, 0]
    angle = torch.atan2(forward[..., 1], forward[..., 0]) / 2
    zero = torch.zeros_like(angle)
    return torch.stack((angle.cos(), zero, zero, angle.sin()), -1)


def orientation(robot_quat, reference_quat, convention):
    frame = heading(robot_quat) if convention == "heading" else robot_quat
    relative = quat_mul(quat_inv(frame[:, None]), reference_quat)
    return matrix(relative)[..., :2].reshape(*relative.shape[:-1], 6)


def g1_features(joint_pos, joint_vel, ref_quat, robot_quat, convention):
    b, frames, _ = joint_pos.shape
    # Preserve SONIC's non-flat chop: NOT per-frame cat(joint_pos, joint_vel).
    chop = torch.cat((joint_pos.reshape(b, -1), joint_vel.reshape(b, -1)), -1).reshape(b, frames, 58)
    return torch.cat((chop, orientation(robot_quat, ref_quat, convention)), -1).reshape(b, -1)


def smpl_features(joints, root_quat, wrist, robot_quat, convention):
    local = torch.matmul(matrix(root_quat).transpose(-1, -2)[:, :, None], joints[..., None]).squeeze(-1)
    return torch.cat((local.flatten(-2), orientation(robot_quat, root_quat, convention), wrist), -1).flatten(1)


def reference_indices(time_steps, clip_ends, frames, skip):
    offsets = torch.arange(frames, device=time_steps.device) * skip
    return torch.minimum(time_steps[:, None] + offsets, clip_ends[:, None] - 1)


def tokenizer(env, command_name="motion", *, version="release", mode="g1"):
    spec = get_spec(version)
    if abs(env.step_dt - spec.dt) > 1e-8:
        raise ValueError(f"{version} requires 50 Hz control/reference data")
    command = env.command_manager.get_term(command_name)
    idx = command.future_frames(spec.frames(mode), spec.g1_skip if mode == "g1" else spec.smpl_skip)
    robot_q = env.scene["robot"].data.root_link_quat_w
    if mode == "g1":
        pelvis = command.cfg.body_names.index("pelvis")
        return g1_features(command.motion.joint_pos[idx], command.motion.joint_vel[idx],
                           command.motion.body_quat_w[idx, pelvis], robot_q, spec.orientation)
    return smpl_features(command.motion.smpl_joints[idx], command.motion.smpl_root_quat[idx],
                         command.motion.joint_pos[idx][:, :, WRIST_MJ], robot_q, spec.orientation)


def observation_groups(version: str, mode: str, command_name="motion"):
    get_spec(version).frames(mode)
    if version == "release":
        from mocke.sonic import profile
        return profile.extra_obs_groups(command_name, mode=mode)
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    return {"tokenizer": ObservationGroupCfg(
        terms={f"{mode}_tokenizer": ObservationTermCfg(func=tokenizer, params={
            "command_name": command_name, "version": version, "mode": mode})},
        concatenate_terms=True, enable_corruption=False)}


def zero_wrist_smpl_tokenizer(env, command_name="motion", **kwargs):
    """Original release tokenizer, with an explicit unavailable-wrist convention."""
    from mocke.sonic.mdp.observations import sonic_smpl_tokenizer
    value = sonic_smpl_tokenizer(env, command_name=command_name, **kwargs)
    frames = value.reshape(env.num_envs, -1, 84).clone()
    frames[..., 78:84] = 0
    return frames.flatten(1)
