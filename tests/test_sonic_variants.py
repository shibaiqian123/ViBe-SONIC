"""Version contracts and pure numerical paths; no GPU, data or simulator required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orcs.core.sonic.specs import IL_ORDER, MJ_ORDER, SPECS, contract_id, seed_path


def test_version_contracts_and_seed_paths(tmp_path):
    assert set(IL_ORDER) == set(MJ_ORDER) and len(MJ_ORDER) == 29
    assert SPECS['release'].width('smpl') == 840
    assert SPECS['v1.1'].width('smpl') == 840
    assert SPECS['low-latency'].width('smpl') == 336
    assert SPECS['low-latency'].g1_skip == 1
    assert SPECS['v1.1'].orientation == 'heading'
    assert seed_path(tmp_path, 'release') == tmp_path / 'seed_state.npz'
    assert seed_path(tmp_path, 'v1.1') == tmp_path / 'seeds/v1.1/seed_state.npz'
    assert len({contract_id(s.contract('smpl'), 'same-weights') for s in SPECS.values()}) == 3


@pytest.fixture
def torch():
    return pytest.importorskip('torch')


def test_reference_sampling_clamps_each_clip(torch):
    from orcs.core.sonic.observations import reference_indices
    times, ends = torch.tensor([0, 9, 21]), torch.tensor([10, 10, 25])
    indices = reference_indices(times, ends, 4, 5)
    assert indices.tolist() == [[0, 5, 9, 9], [9, 9, 9, 9], [21, 24, 24, 24]]


def test_layout_permutations_are_bijective(torch):
    from orcs.core.sonic.port import input_layout, input_permutation
    for mode in ('g1', 'policy'):
        p = input_permutation(mode)
        assert p.unique().numel() == p.numel()
        assert torch.equal(input_layout(MJ_ORDER, mode)[p], input_layout(IL_ORDER, mode))


def test_dynamic_mlp_layer_extraction(torch):
    from orcs.core.sonic.port import extract_mlp
    state = {'enc.0.weight': torch.randn(8, 4), 'enc.0.bias': torch.zeros(8),
             'enc.2.weight': torch.randn(3, 8), 'enc.2.bias': torch.zeros(3)}
    _, dims = extract_mlp(state, 'enc.')
    assert dims == (4, 8, 3)
    state['enc.running_mean'] = torch.zeros(4)
    with pytest.raises(ValueError, match='unsupported'):
        extract_mlp(state, 'enc.')


def test_heading_and_smpl_features_match_independent_rotations(torch):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from orcs.core.sonic.observations import g1_features, smpl_features
    rng = np.random.default_rng(71)
    # Include yaw discontinuity, nonzero roll/pitch, and quaternion sign flips.
    angles = np.array([[.3, -.2, np.pi-1e-6], [-.3, .4, -np.pi+1e-6]])
    robot_rot = Rotation.from_euler('xyz', angles)
    q = robot_rot.as_quat()[:, [3, 0, 1, 2]]
    q[1] *= -1
    for spec in SPECS.values():
        for mode in ('g1', 'smpl'):
            f = spec.frames(mode)
            ref = Rotation.random(2*f, random_state=rng)
            ref_q = ref.as_quat()[:, [3, 0, 1, 2]].reshape(2, f, 4)
            frame = Rotation.from_euler('z', angles[:, 2]) if spec.orientation == 'heading' else robot_rot
            relative = np.einsum('bij,bfjk->bfik', frame.as_matrix().transpose(0, 2, 1), ref.as_matrix().reshape(2, f, 3, 3))
            ori = relative[..., :2].reshape(2, f, 6)
            def tensor(a):
                return torch.tensor(a, dtype=torch.float32)
            if mode == 'g1':
                jp, jv = rng.normal(size=(2, f, 29)), rng.normal(size=(2, f, 29))
                expected = np.concatenate([np.concatenate([jp.reshape(2, -1), jv.reshape(2, -1)], -1).reshape(2, f, 58), ori], -1).reshape(2, -1)
                actual = g1_features(tensor(jp), tensor(jv), tensor(ref_q), tensor(q), spec.orientation)
            else:
                joints, wrist = rng.normal(size=(2, f, 24, 3)), rng.normal(size=(2, f, 6))
                local = np.einsum('bfji,bfkj->bfki', ref.as_matrix().reshape(2, f, 3, 3), joints)
                expected = np.concatenate([local.reshape(2, f, 72), ori, wrist], -1).reshape(2, -1)
                actual = smpl_features(tensor(joints), tensor(ref_q), tensor(wrist), tensor(q), spec.orientation)
            np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("constructor_style", ["keyword", "runner_positional"])
def test_zero_adapter_updates_only_trainable_weights_and_resume_guard(torch, tmp_path, constructor_style):
    pytest.importorskip('rsl_rl.models')
    from tensordict import TensorDict

    from orcs.core.sonic.model import VersionedSonicWithAdapterModel
    from orcs.core.sonic.port import make_mlp
    torch.manual_seed(21)
    e, d = make_mlp((64, 16, 64)), make_mlp((994, 16, 29))
    path = tmp_path / 'tiny.pt'
    torch.save({'model_state_dict': {**{'encoder.'+k:v for k,v in e.state_dict().items()},
                                     **{'decoder.'+k:v for k,v in d.state_dict().items()}},
                'meta': {'action_std': torch.full((29,), .1)}}, path)
    obs = TensorDict({'policy': torch.randn(3, 930), 'tokenizer': torch.randn(3, 64),
                      'augmentation': torch.randn(3, 7)}, batch_size=[3])
    kw = dict(obs=obs, obs_groups={'actor':['policy']}, obs_set='actor', output_dim=29,
              sonic_contract_id='ab'*32, tokenizer_width=64, encoder_hidden_dims=[16], decoder_hidden_dims=[16],
              base_checkpoint=str(path), freeze_base=True, rank=2, alpha=2.)
    if constructor_style == "runner_positional":
        positional = [kw.pop(key) for key in ("obs", "obs_groups", "obs_set", "output_dim")]
        model = VersionedSonicWithAdapterModel(*positional, **kw)
    else:
        model = VersionedSonicWithAdapterModel(**kw)
    from orcs.core.sonic.port import fsq
    expected = d(torch.cat([fsq(e(obs['tokenizer'])), obs['policy']], -1))
    actual = model(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    trainable = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    model(obs).square().mean().backward()
    optimizer.step()
    assert frozen and trainable
    assert all(torch.equal(p, frozen[n]) for n,p in model.named_parameters() if n in frozen)
    assert any(not torch.equal(p, trainable[n]) for n,p in model.named_parameters() if n in trainable)
    state = model.state_dict()
    model.load_state_dict(state)
    legacy = {k:v for k,v in state.items() if k != '_orcs_sonic_contract'}
    with pytest.raises(ValueError, match='refusing checkpoint resume'):
        model.load_state_dict(legacy)
    model._allow_legacy_resume = True
    model.load_state_dict(legacy)
    frozen_key = next(iter(frozen))
    broken = {**legacy, frozen_key: legacy[frozen_key] + 1}
    with pytest.raises(ValueError, match='frozen weights'):
        model.load_state_dict(broken)
    wrong = {**state, '_orcs_sonic_contract': torch.zeros(32, dtype=torch.uint8)}
    with pytest.raises(ValueError, match='refusing checkpoint resume'):
        model.load_state_dict(wrong)


def test_seed_version_receipts_do_not_cross(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "src/orcs/core/data/seed_receipt.py"
    spec = importlib.util.spec_from_file_location("isolated_seed_receipt", path)
    receipt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receipt_module)
    seed_matches_source, source_digest = receipt_module.seed_matches_source, receipt_module.source_digest
    sample = tmp_path / 'sample0'
    sample.mkdir()
    (sample/'smpl_motion.npz').write_bytes(b'data')
    (sample/'metadata.json').write_text('{}')
    (tmp_path/'tile.json').write_text('{}')
    p = seed_path(sample, 'v1.1').with_name('seed_source.json')
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'source_sha256': source_digest(sample), 'sonic_version': 'v1.1', 'contract_id': 'v11'}))
    assert seed_matches_source(sample, 'v1.1', 'v11')
    assert not seed_matches_source(sample, 'low-latency', 'v11')
    assert not seed_matches_source(sample, 'v1.1', 'another-weight')


def test_native_converter_rejects_wrong_file_before_unpickling(torch, tmp_path, monkeypatch):
    from orcs.core.sonic import port
    checkpoint = tmp_path / 'last.pt'
    checkpoint.write_bytes(b'wrong checkpoint')
    def must_not_load(path):
        raise AssertionError('unverified checkpoint must not be unpickled')
    monkeypatch.setattr(port, 'load_official', must_not_load)
    with pytest.raises(ValueError, match='SHA-256'):
        port.convert(checkpoint, tmp_path / 'output', 'v1.1', True)
    assert not (tmp_path / 'output').exists()
