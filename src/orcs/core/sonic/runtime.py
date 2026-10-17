"""Bind an opt-in task to one immutable SONIC model/observation contract."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from .specs import ENCODER_HIDDEN, MJ_ORDER, checkpoint_path, contract_id, get_spec


@lru_cache(maxsize=16)
def _metadata(path: str, mtime: int, size: int) -> dict:
    import torch
    del mtime, size
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = dict(payload['meta'])
    meta.pop('action_std', None)
    return meta


def model_info(version: str, mode: str) -> tuple[Path, dict]:
    spec = get_spec(version)
    path = checkpoint_path(version, mode)
    # Legacy release behavior is preserved when native versioned weights are absent.
    if version == 'release' and not path.is_file():
        from orcs.core.rl import SMPL_CKPT, SONIC_CKPT
        path = Path(SMPL_CKPT if mode == 'smpl' else SONIC_CKPT)
    if not path.is_file():
        raise FileNotFoundError(f"missing {version}/{mode} checkpoint: {path}; run "
                                f"orcs-sonic-checkpoint --version {version} --convert --smpl")
    stat = path.stat()
    meta = _metadata(str(path), stat.st_mtime_ns, stat.st_size)
    contract = spec.contract(mode)
    if 'sonic_version' in meta:
        if meta['sonic_version'] != version or meta.get('contract') != contract:
            raise ValueError(f"checkpoint version/observation contract mismatch: {path}")
        if meta.get('contract_id') != contract_id(contract, meta['source_sha256']):
            raise ValueError(f"checkpoint contract fingerprint mismatch: {path}")
    elif version != 'release':
        raise ValueError(f"unversioned checkpoint cannot be used as {version}")
    else:
        if meta.get('tokenizer_dim') != spec.width(mode):
            raise ValueError(f"legacy release tokenizer mode mismatch: {path}")
        meta.update(sonic_version='release', encoder_mode=mode, contract=contract,
                    encoder_hidden_dims=list(ENCODER_HIDDEN), decoder_hidden_dims=list(spec.decoder_hidden))
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                digest.update(chunk)
        meta['source_sha256'] = digest.hexdigest()
        meta['contract_id'] = contract_id(contract, meta['source_sha256'])
    return path, meta


def configure_env(cfg, version: str, mode: str):
    from .observations import observation_groups
    _, meta = model_info(version, mode)
    cfg.observations.update(observation_groups(version, mode))
    command = cfg.commands['motion']
    from orcs.core.data.seed_identity import STAIR_POINT_MAPPING
    command.point_mapping = STAIR_POINT_MAPPING
    command.sonic_version = version
    command.sonic_contract_id = meta['contract_id']
    # Runtime metadata on config survives serialized experiment snapshots.
    return cfg


def agent_cfg(version: str, mode: str, experiment_name: str):
    from orcs.core.rl import adapt_sonic_agent_cfg
    path, meta = model_info(version, mode)
    cfg = adapt_sonic_agent_cfg(experiment_name, base_checkpoint=str(path))
    cfg.actor.update(class_name='orcs.core.sonic.model.VersionedSonicWithAdapterModel',
                     encoder_hidden_dims=meta['encoder_hidden_dims'],
                     decoder_hidden_dims=meta['decoder_hidden_dims'],
                     sonic_contract_id=meta['contract_id'],
                     tokenizer_width=get_spec(version).width(mode),
                     adapt_encoder=False, adapt_decoder=True, allow_legacy_resume=(version == "release"))
    return cfg


def check_robot_order(robot) -> None:
    if tuple(robot.joint_names) != MJ_ORDER:
        raise ValueError('versioned SONIC requires the verified G1 MuJoCo joint order')
