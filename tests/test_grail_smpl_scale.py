"""GRAIL human scale affects both reference channels around authored trans."""
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/orcs/tasks/perloco/sources/smplx_fk.py'
spec = importlib.util.spec_from_file_location('grail_fk', SOURCE)
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)


def test_scale_pivot_is_translation_not_pelvis_or_world_origin():
    trans = np.array([[10., 20., 30.]], dtype=np.float32)
    joints = trans[:, None] + np.array([[[1., 2., 3.], [4., 5., 6.]]], dtype=np.float32)
    before = joints.copy()
    result = fk._scaled_world_joints(joints, trans, [0.5])
    np.testing.assert_array_equal(result, [[[10.5, 21., 31.5], [12., 22.5, 33.]]])
    np.testing.assert_array_equal(joints, before)


def test_default_scale_is_bit_exact():
    joints = np.random.default_rng(1).normal(size=(3, 24, 3)).astype(np.float32)
    trans = np.ones((3, 3), dtype=np.float32) * 10000
    np.testing.assert_array_equal(fk._scaled_world_joints(joints, trans), joints)


@pytest.mark.parametrize('scale', [0., -1., np.nan, np.inf, [1., 2.]])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match='finite positive scalar'):
        fk._scaled_world_joints(np.zeros((2, 24, 3)), np.zeros((2, 3)), scale)


def test_channel_pipeline_scales_world_and_encoder_but_not_orientation(monkeypatch):
    trans = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    raw = np.arange(2*55*3, dtype=np.float32).reshape(2, 55, 3) / 100 + trans[:, None]
    rec = dict(poses=np.zeros((2, 165), dtype=np.float32), trans=trans, betas=np.zeros((1, 10)))
    monkeypatch.setitem(sys.modules, 'joblib', SimpleNamespace(load=lambda _: {'human_data':rec}))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        as_tensor=lambda x:x, zeros=lambda *shape:np.zeros(shape), no_grad=nullcontext))

    class Model:
        num_betas = 10
        num_expression_coeffs = 10

        def __call__(self, **kwargs):
            return SimpleNamespace(joints=SimpleNamespace(numpy=lambda:raw.copy()))

    monkeypatch.setattr(fk, '_model', lambda *args:Model())
    joints0, quat0, viz0 = fk.smpl_channels('unused', 'unused')
    rec['scale'] = np.array([0.76332754], dtype=np.float32)
    joints, quat, viz = fk.smpl_channels('unused', 'unused')
    np.testing.assert_allclose(viz, (viz0-trans[:,None])*rec['scale'][0]+trans[:,None])
    np.testing.assert_allclose(joints, joints0*rec['scale'][0], atol=1e-6)
    np.testing.assert_array_equal(quat, quat0)
