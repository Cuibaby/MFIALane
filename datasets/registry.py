from utils import Registry, build_from_cfg

import torch

DATASETS = Registry('datasets')
PROCESS = Registry('process')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dataset(split_cfg, cfg):
    args = split_cfg.copy()
    args.pop('type')
    args = args.to_dict()
    args['cfg'] = cfg
    return build(split_cfg, DATASETS, default_args=args)

def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(split_cfg, cfg)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = cfg.batch_size, shuffle = shuffle,
        num_workers = cfg.workers, pin_memory = False, drop_last = False)

    return data_loader
