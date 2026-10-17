"""Convert official checkpoints to explicit MJLab-order, versioned contracts.

This is intentionally separate from model construction and never downloads weights.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path

import torch

from .specs import (
    ENCODER_HIDDEN,
    IL_ORDER,
    MJ_ORDER,
    OFFICIAL_SHA256,
    contract_id,
    get_spec,
)


def load_official(path: Path) -> dict:
    """Read trusted official weights, preserving obsolete config objects as metadata."""
    class ConfigPlaceholder:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(("groot.", "gear_sonic.", "trl.")):
                return type(name, (ConfigPlaceholder,), {"__module__": module})
            return super().find_class(module, name)

    class PickleModule:
        pass
    PickleModule.Unpickler = Unpickler
    PickleModule.load = staticmethod(pickle.load)
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=PickleModule, mmap=True)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_mlp(policy: dict, prefix: str) -> tuple[dict, tuple[int, ...]]:
    state = {key[len(prefix):]: value.detach().float().cpu() for key, value in policy.items() if key.startswith(prefix)}
    if not state:
        raise ValueError(f"missing checkpoint prefix {prefix}")
    if any(not re.fullmatch(r"\d+\.(weight|bias)", key) for key in state):
        raise ValueError(f"unsupported MLP parameters in {prefix}: {list(state)}")
    indices = sorted(int(key.split('.')[0]) for key in state if key.endswith('.weight'))
    if indices != list(range(0, 2 * len(indices), 2)):
        raise ValueError(f"unexpected MLP layer numbering in {prefix}: {indices}")
    dims = []
    for index in indices:
        w, b = state[f"{index}.weight"], state.get(f"{index}.bias")
        if w.ndim != 2 or b is None or b.shape != (w.shape[0],):
            raise ValueError(f"malformed linear layer {prefix}{index}")
        if dims and dims[-1] != w.shape[1]:
            raise ValueError(f"disconnected MLP shape {prefix}{index}")
        if not dims:
            dims.append(w.shape[1])
        dims.append(w.shape[0])
    if len(state) != 2 * len(indices):
        raise ValueError(f"unhandled parameters in {prefix}")
    return state, tuple(dims)


def make_mlp(dims):
    layers = []
    for index, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(torch.nn.Linear(a, b))
        if index != len(dims) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def input_layout(order, mode, frames=10):
    slots = torch.tensor([MJ_ORDER.index(name) for name in order])
    if mode == "policy":
        history = torch.arange(10)[:, None]
        return torch.cat(
            [3000+torch.arange(30)] + [(base+history*29+slots).flatten() for base in (4000, 5000, 6000)]
            + [7000+torch.arange(30)])
    t = torch.arange(frames)[:, None]
    chopped = torch.cat([(t*29+slots).flatten(), (1000+t*29+slots).flatten()]).reshape(frames, 58)
    ori = 2000 + torch.arange(frames*6).reshape(frames, 6)
    return torch.cat([chopped, ori], -1).flatten()


def input_permutation(mode, frames=10):
    theirs, ours = input_layout(IL_ORDER, mode, frames), input_layout(MJ_ORDER, mode, frames)
    result = torch.empty_like(theirs)
    result[torch.argsort(theirs)] = torch.argsort(ours)
    if not torch.equal(ours[result], theirs):
        raise ValueError("joint layout is not a permutation")
    return result


def fsq(latent):
    levels = torch.full((32,), 32, dtype=torch.long, device=latent.device)
    half = (levels - 1) * (1 + 1e-3) / 2
    offset = torch.where(levels % 2 == 0, 0.5, 0.0)
    shift = (offset / half).atanh()
    shaped = latent.reshape(*latent.shape[:-1], 2, 32)
    bounded = (shaped + shift).tanh() * half - offset
    quantized = bounded + (bounded.round() - bounded).detach()
    return (quantized / (levels // 2)).reshape_as(latent)


def remap_decoder(decoder: dict) -> dict:
    actor_order = torch.tensor([MJ_ORDER.index(n) for n in IL_ORDER])
    permutation = torch.cat([torch.arange(64), 64+input_permutation("policy")])
    result = {k: v.clone() for k, v in decoder.items()}
    result['0.weight'][:, permutation] = decoder['0.weight']
    last = max(int(k.split('.')[0]) for k in decoder)
    result[f'{last}.weight'][actor_order] = decoder[f'{last}.weight']
    result[f'{last}.bias'][actor_order] = decoder[f'{last}.bias']
    return result


def remap_encoder(encoder: dict, mode: str, frames: int) -> tuple[dict, torch.Tensor]:
    result = {k: v.clone() for k, v in encoder.items()}
    permutation = input_permutation('g1', frames) if mode == 'g1' else torch.arange(encoder['0.weight'].shape[1])
    result['0.weight'][:, permutation] = encoder['0.weight']
    return result, permutation


def convert(path: Path, out_dir: Path, version: str, smpl: bool = False) -> dict:
    spec = get_spec(version)
    sha = file_hash(path)
    if sha != OFFICIAL_SHA256[version]:
        raise ValueError(f"{version}: checkpoint SHA-256 does not match the pinned official release")
    checkpoint = load_official(path)
    policy = checkpoint.get("policy_state_dict")
    if not isinstance(policy, dict):
        raise ValueError("checkpoint must contain policy_state_dict")
    accepted = ("actor_module.encoders.g1.module.", "actor_module.encoders.smpl.module.",
                "actor_module.encoders.teleop.module.", "actor_module.decoders.g1_dyn.module.",
                "actor_module.decoders.g1_kin.module.", "critic_module.")
    unknown = [k for k in policy if not k.startswith(accepted) and k != "std"]
    if unknown:
        raise ValueError(f"unhandled checkpoint parameters (normalizers/quantizer included): {unknown[:20]}")
    decoder, dec_dims = extract_mlp(policy, "actor_module.decoders.g1_dyn.module.")
    if dec_dims != (994, *spec.decoder_hidden, 29):
        raise ValueError(f"{version}: decoder dimensions {dec_dims} differ from the official contract")
    std = policy.get("std")
    if std is None or std.shape != (29,) or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("missing/invalid action std")
    actor_order = torch.tensor([MJ_ORDER.index(n) for n in IL_ORDER])
    dec_mj = remap_decoder(decoder)
    std_mj = torch.empty_like(std)
    std_mj[actor_order] = std
    report = {"version": version, "source_sha256": sha, "decoder_dims": dec_dims, "modes": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in (('g1', 'smpl') if smpl else ('g1',)):
        encoder, enc_dims = extract_mlp(policy, f"actor_module.encoders.{mode}.module.")
        if enc_dims != (spec.width(mode), *ENCODER_HIDDEN, 64):
            raise ValueError(f"{version}/{mode}: encoder shape {enc_dims} violates contract")
        enc_mj, enc_perm = remap_encoder(encoder, mode, spec.g1_frames)
        # Compare source and remapped networks before publishing. No adapter involved.
        e0, e1 = make_mlp(enc_dims), make_mlp(enc_dims)
        d0, d1 = make_mlp(dec_dims), make_mlp(dec_dims)
        e0.load_state_dict(encoder)
        e1.load_state_dict(enc_mj)
        d0.load_state_dict(decoder)
        d1.load_state_dict(dec_mj)
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(8, enc_dims[0], generator=generator)
        prop = torch.randn(8, 930, generator=generator)
        with torch.inference_mode():
            z0, z1 = e0(x[:, enc_perm]), e1(x)
            torch.testing.assert_close(z0, z1, atol=1e-4, rtol=1e-4)
            t0, t1 = fsq(z0), fsq(z1)
            mismatch = (t0 != t1).any(-1)
            # Discontinuities require separate reporting, not relaxed action tolerances.
            good = ~mismatch
            if not good.any():
                raise ValueError("all parity samples crossed quantization boundaries")
            y0 = d0(torch.cat([t0[good], prop[good][:, input_permutation('policy')]], -1))
            expected = torch.empty_like(y0)
            expected[:, actor_order] = y0
            actual = d1(torch.cat([t1[good], prop[good]], -1))
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        contract = spec.contract(mode)
        meta = {**contract, "sonic_version": version, "encoder_hidden_dims": list(enc_dims[1:-1]),
                "decoder_hidden_dims": list(dec_dims[1:-1]), "action_std": std_mj,
                "source_sha256": sha, "source": str(path), "contract": contract,
                "contract_id": contract_id(contract, sha), "num_future_frames": spec.frames(mode),
                "future_dt": spec.dt*(spec.g1_skip if mode == 'g1' else spec.smpl_skip),
                "history_length": 10}
        payload = {"model_state_dict": {**{'encoder.'+k:v for k,v in enc_mj.items()},
                                         **{'decoder.'+k:v for k,v in dec_mj.items()}}, "meta": meta}
        destination = out_dir / ('last_ported.pt' if mode == 'g1' else 'smpl_ported.pt')
        if destination.exists():
            raise FileExistsError(f"refusing overwrite: {destination}")
        torch.save(payload, destination)
        report['modes'][mode] = {"encoder_dims": enc_dims, "latent_max_error": (z0-z1).abs().max().item(),
                                 "action_max_error": (actual-expected).abs().max().item(),
                                 "boundary_samples": int(mismatch.sum()), "tested_samples": len(x)}
        del e0, e1, d0, d1
    (out_dir / 'parity.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--smpl', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(json.dumps(convert(a.ckpt, a.out_dir, a.version, a.smpl), indent=2))


if __name__ == '__main__':
    main()
