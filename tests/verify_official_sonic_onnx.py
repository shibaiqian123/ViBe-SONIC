"""Server-only cross-check of ORCS permutations against existing official ONNX.

This checks the actual published networks without substituting ONNX for a training
checkpoint. It does not create ported weights or mark a last.pt as verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from orcs.core.sonic.port import (
    fsq,
    make_mlp,
    remap_decoder,
    remap_encoder,
)
from orcs.core.sonic.specs import IL_ORDER, MJ_ORDER, SPECS


def check(root: Path, version: str):
    spec = SPECS[version]
    folder = root / spec.directory
    encoder_file = folder / 'model_encoder.onnx'
    decoder_file = folder / 'model_decoder.onnx'
    encoder_proto = onnx.load(str(encoder_file))
    decoder_proto = onnx.load(str(decoder_file))
    initializers = {v.name: np.array(numpy_helper.to_array(v), copy=True) for v in encoder_proto.graph.initializer}
    decoder_arrays = {v.name: np.array(numpy_helper.to_array(v), copy=True) for v in decoder_proto.graph.initializer}
    dec = {}
    # Export uses MatMul(weight.T) + Add(bias), preserving topological order.
    layers = [n for n in decoder_proto.graph.node if n.op_type in ('MatMul', 'Gemm') and
              len(n.input) > 1 and n.input[1] in decoder_arrays and decoder_arrays[n.input[1]].ndim == 2]
    for index, node in enumerate(layers):
        weight = decoder_arrays[node.input[1]]
        if node.op_type == 'MatMul':
            weight = weight.T
            add = next(n for n in decoder_proto.graph.node if n.op_type == 'Add' and node.output[0] in n.input)
            bias = decoder_arrays[next(x for x in add.input if x != node.output[0])]
        else:
            trans = next((x.i for x in node.attribute if x.name == 'transB'), 0)
            weight = weight if trans else weight.T
            bias = decoder_arrays[node.input[2]]
        dec[f'{2*index}.weight'] = torch.tensor(weight)
        dec[f'{2*index}.bias'] = torch.tensor(bias)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    enc_session = ort.InferenceSession(str(encoder_file), session_options, providers=['CPUExecutionProvider'])
    dec_session = ort.InferenceSession(str(decoder_file), session_options, providers=['CPUExecutionProvider'])
    decoder = make_mlp((994, *spec.decoder_hidden, 29))
    decoder.load_state_dict(remap_decoder(dec))
    rng = np.random.default_rng(31)
    modes = {}
    for mode in ('g1', 'smpl'):
        state = {k.replace('module.encoders.'+mode+'.module.', ''): torch.tensor(v)
                 for k, v in initializers.items() if k.startswith('module.encoders.'+mode+'.module.')}
        mapped, perm = remap_encoder(state, mode, spec.g1_frames)
        encoder = make_mlp((spec.width(mode), 2048, 1024, 512, 512, 64))
        encoder.load_state_dict(mapped)
        token_errors, action_errors = [], []
        for _ in range(8):
            raw = rng.normal(0, .3, (1, enc_session.get_inputs()[0].shape[1])).astype(np.float32)
            raw[0, :4] = [0 if mode == 'g1' else 2, 0, 0, 0]
            if mode == 'g1':
                pos_vel = raw[:, 4:584].reshape(1, 10, 58)
                ori_start = 590 if version == 'low-latency' else 584
                ori = raw[:, ori_start:ori_start+60].reshape(1, 10, 6)
                il_features = np.concatenate((pos_vel, ori), -1).reshape(1, -1)
            else:
                start = 4+640+267
                f = spec.smpl_frames
                joints = raw[:, start:start+f*72].reshape(1, f, 72)
                orientation = raw[:, start+f*72:start+f*78].reshape(1, f, 6)
                wrists = raw[:, start+f*78:start+f*84].reshape(1, f, 6)
                il_features = np.concatenate((joints, orientation, wrists), -1).reshape(1, -1)
            features = torch.empty(1, spec.width(mode))
            features[:, perm] = torch.tensor(il_features)
            with torch.inference_mode():
                tokens = fsq(encoder(features)).numpy()
            official_tokens = enc_session.run(None, {enc_session.get_inputs()[0].name: raw})[0]
            token_errors.append(float(np.max(np.abs(tokens-official_tokens))))
            np.testing.assert_allclose(tokens, official_tokens, atol=1e-4, rtol=1e-4)
            prop_mj = torch.tensor(rng.normal(0, .2, (1, 930)).astype(np.float32))
            from orcs.core.sonic.port import input_permutation
            official_in = np.concatenate((official_tokens, prop_mj[:, input_permutation('policy')].numpy()), -1)
            original_action = dec_session.run(None, {dec_session.get_inputs()[0].name: official_in})[0]
            expected = np.empty_like(original_action)
            expected[:, [MJ_ORDER.index(n) for n in IL_ORDER]] = original_action
            with torch.inference_mode():
                actual = decoder(torch.cat((torch.tensor(tokens), prop_mj), -1)).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            action_errors.append(float(np.max(np.abs(actual-expected))))
        modes[mode] = {'samples': 8, 'token_max_error': max(token_errors), 'action_max_error': max(action_errors)}
    return {'version':version, 'source':'official ONNX, not last.pt conversion', 'modes':modes,
            'onnx_orientation_layout': 'single_then_multi (differs from published observation YAML)' if version == 'low-latency' else 'multi_then_single'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--versions', nargs='+', default=['v1.1','low-latency'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = [check(args.root, v) for v in args.versions]
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
