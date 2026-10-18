"""Opt-in human-clock wiring shared by seed synthesis and SMPL training."""
import json

from orcs.core.data.human_timeline import CONTRACT, validate


def configure(cfg, root, sonic_version="release"):
    samples = sorted(root.glob("*/level_*/sample*/metadata.json"))
    flags = [json.loads(p.read_text()).get("timeline_contract") == CONTRACT for p in samples]
    if not any(flags):
        return cfg
    if sonic_version != "release":
        raise ValueError("human-clock baseline currently supports SONIC release only")
    if not all(flags):
        raise ValueError("cannot mix human-clock and old robot-clock stair samples")
    for p in samples:
        validate(p.parent)
    cfg.commands["motion"].point_mapping = "legacy"
    from orcs.core.sonic.observations import zero_wrist_smpl_tokenizer
    group = cfg.observations["tokenizer"]
    if len(group.terms) != 1:
        raise ValueError("unexpected SMPL tokenizer group")
    term = next(iter(group.terms.values()))
    term.func = zero_wrist_smpl_tokenizer
    return cfg
