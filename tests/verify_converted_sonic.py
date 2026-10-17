"""Server acceptance for real converted weights (not a physics/PPO smoke test)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rsl_rl.models import SonicBaseModel
from tensordict import TensorDict

from orcs.core.sonic.model import VersionedSonicWithAdapterModel


def check(path: Path):
    meta = torch.load(path, map_location='cpu', weights_only=True, mmap=True)['meta']
    obs = TensorDict({'policy': torch.randn(2, 930), 'tokenizer': torch.randn(2, meta['tokenizer_dim']),
                      'augmentation': torch.randn(2, 53)}, batch_size=[2])
    shared = dict(obs=obs, obs_groups={'actor':['policy']}, obs_set='actor', output_dim=29,
                  encoder_hidden_dims=meta['encoder_hidden_dims'], decoder_hidden_dims=meta['decoder_hidden_dims'],
                  base_checkpoint=str(path), freeze_base=True)
    base = SonicBaseModel(**shared)
    adapted = VersionedSonicWithAdapterModel(**shared, sonic_contract_id=meta['contract_id'],
                                             tokenizer_width=meta['tokenizer_dim'], rank=16, alpha=1.)
    with torch.no_grad():
        before, after = base(obs), adapted(obs)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    del base
    frozen = {n:p.detach().clone() for n,p in adapted.named_parameters() if not p.requires_grad}
    trainable = {n:p.detach().clone() for n,p in adapted.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([p for p in adapted.parameters() if p.requires_grad], lr=1e-4)
    loss = adapted(obs).square().mean()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(p, frozen[n]) for n,p in adapted.named_parameters() if n in frozen)
    assert any(not torch.equal(p, trainable[n]) for n,p in adapted.named_parameters() if n in trainable)
    state = adapted.state_dict()
    adapted.load_state_dict(state)
    wrong = dict(state)
    wrong['_orcs_sonic_contract'] = state['_orcs_sonic_contract'].clone()
    wrong['_orcs_sonic_contract'][0] ^= 1
    try:
        adapted.load_state_dict(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError('cross-contract resume was accepted')
    return {'version':meta['sonic_version'], 'mode':meta['encoder_mode'],
            'zero_adapter_max_error':float((before-after).abs().max()),
            'frozen_parameters_unchanged':True, 'adapter_updated':True,
            'same_contract_restore':True, 'wrong_contract_rejected':True,
            'source_sha256':meta['source_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(19)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {'models': [check(args.root / name) for name in ('last_ported.pt','smpl_ported.pt')],
              'physics_and_ppo': 'not_run: requires staged stairs, valid seeds and complete simulator/assets'}
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
