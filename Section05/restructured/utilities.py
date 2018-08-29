import json


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_config(config_path):
    config = None
    with open(config_path) as f:
        config = json.load(f)

    return AttrDict(config)
