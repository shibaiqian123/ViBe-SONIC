"""Contract behavior without datasets or simulator initialization."""
import importlib.util
import sys
import types
from pathlib import Path
import pytest
import torch


def test_shared_contract(monkeypatch):
    class Base(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    fake = types.ModuleType("rsl_rl.models")
    fake.SonicWithAdapterModel = Base
    monkeypatch.setitem(sys.modules, "rsl_rl.models", fake)
    path = Path(__file__).parents[1] / "src/orcs/core/sonic/model.py"
    spec = importlib.util.spec_from_file_location("contract_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obs = {"tokenizer": torch.zeros(2,840), "policy": torch.zeros(2,930)}
    def model(identity="01"*32, **kw):
        return module.VersionedSonicWithAdapterModel(obs, {}, "actor", 29,
            sonic_contract_id=identity, tokenizer_width=840, **kw)
    a, b = model(), model()
    b.load_state_dict(a.state_dict())
    with pytest.raises(ValueError, match="refusing checkpoint"):
        model("02"*32).load_state_dict(a.state_dict())
    with pytest.raises(ValueError, match="refusing checkpoint"):
        b.load_state_dict({"weight": torch.ones(2)})
    model(allow_legacy_resume=True).load_state_dict({"weight": torch.ones(2)})
    with pytest.raises(ValueError, match="frozen weights"):
        model(allow_legacy_resume=True).load_state_dict({"weight": torch.zeros(2)})
    obs["tokenizer"] = torch.zeros(2,640)
    with pytest.raises(ValueError, match="dimensions"):
        model()
