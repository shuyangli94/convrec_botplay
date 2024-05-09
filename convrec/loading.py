import torch
from convrec.utils import load_checkpoint, load_config
from convrec import system
from omegaconf import OmegaConf


def load_model(ckpt_path: str,
               cfg_path: str = None,
               cfg: OmegaConf = None,
               strict: bool = True):
    assert ((cfg_path is not None) or (cfg is not None))

    if cfg is None:
        cfg = load_config(cfg_path)

    SystemClass = getattr(system, cfg.system)
    model = SystemClass(cfg).model

    # Load state dict
    state_dict = load_checkpoint(ckpt_path)
    model.load_state_dict(state_dict, strict=strict)

    return cfg, model


def load_agent_states(ckpt_path):
    raw_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    agent_state_dict = dict()
    model_state_dict = dict()
    for k, v in raw_state_dict.items():
        if k.startswith('model.'):
            model_state_dict[k.replace('model.', '')] = v
        elif k.startswith('agent.'):
            agent_state_dict[k.replace('agent.', '')] = v
    return model_state_dict, agent_state_dict


def load_agent(ckpt_path: str,
               cfg_path: str = None,
               cfg: OmegaConf = None,
               strict: bool = True):
    assert ((cfg_path is not None) or (cfg is not None))

    if cfg is None:
        cfg = load_config(cfg_path)

    SystemClass = getattr(system, cfg.system)
    system_obj = SystemClass(cfg)
    model = system_obj.model
    agent = system_obj.agent

    # Load state dict
    model_state, agent_state = load_agent_states(ckpt_path)
    model.load_state_dict(model_state, strict=True)
    agent.load_state_dict(agent_state, strict=True)

    return cfg, model, agent
