"""Model-level compatibility guard also applies when the runner restores a checkpoint."""
from __future__ import annotations

import torch
from rsl_rl.models import SonicWithAdapterModel


class SonicContractMixin:
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, sonic_contract_id: str,
                 tokenizer_width: int, allow_legacy_resume: bool = False, **kwargs):
        if obs['tokenizer'].shape[-1] != tokenizer_width or obs['policy'].shape[-1] != 930:
            raise ValueError('SONIC observation dimensions do not match selected checkpoint')
        self._allow_legacy_resume = allow_legacy_resume
        self._loading_base = True
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self._loading_base = False
        identity = torch.tensor(list(bytes.fromhex(sonic_contract_id)), dtype=torch.uint8)
        self.register_buffer('_orcs_sonic_contract', identity)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not getattr(self, '_loading_base', False):
            incoming = state_dict.get('_orcs_sonic_contract')
            if incoming is None and self._allow_legacy_resume:
                # Old release checkpoints lack a version tag. Accept only when every
                # frozen parameter is bit-identical to the selected release base.
                frozen = {name: value for name, value in self.named_parameters() if not value.requires_grad}
                if not frozen or any(name not in state_dict or not torch.equal(
                        value.detach().cpu(), state_dict[name].detach().cpu()) for name, value in frozen.items()):
                    raise ValueError('legacy checkpoint frozen weights do not match selected release')
                state_dict = dict(state_dict)
                state_dict['_orcs_sonic_contract'] = self._orcs_sonic_contract.clone()
                incoming = state_dict['_orcs_sonic_contract']
            if incoming is None or not torch.equal(incoming.cpu(), self._orcs_sonic_contract.cpu()):
                raise ValueError('refusing checkpoint resume: SONIC version/mode/base weights differ')
        return super().load_state_dict(state_dict, strict=strict, **kwargs)


class VersionedSonicWithAdapterModel(SonicContractMixin, SonicWithAdapterModel):
    pass
