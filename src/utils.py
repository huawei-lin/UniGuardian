import numpy as np
import random
import json
import collections.abc

class Struct:
    """The recursive class for building and representing objects with."""

    def __init__(self, obj={}, **kwargs):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return "{%s}" % str(
            ", ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )

def get_default_config():
    """Returns a default config file"""
    config = {
        "model_name_or_path": None,
        "dataset_path": None,
        "output_path": "./output.jsonl",
        "system_prompt_path": "",
        "system_prompt": "",
        "generate": {
            "temperature": 1,
            "max_tokens": 256,
            "min_tokens": 16,
        },
        "detection": {
            "num_masks_per_instruction": 0.5,
            "max_masks_per_instruction": 8,
            "num_masked_instructions": 100,
        }
    }
    return config


def sanity_check(config):
    if not config.dataset_path:
        raise Exception('Missing "dataset_path"')
    if not config.model_name_or_path:
        raise Exception('Missing "model_name_or_path"')

    system_prompt = config.system_prompt
    if system_prompt is None or len(system_prompt) == 0:
        system_prompt_path = config.system_prompt_path
        if system_prompt_path is not None and len(system_prompt_path) > 0:
            with open(system_prompt_path, "r") as fr:
                config.system_prompt = fr.read()
            print("system_prompt:", config.system_prompt)


def get_config(config_path):
    """Returns a  config file"""

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = get_default_config()
    config = update(config, json.load(open(config_path)))
    config = Struct(config)
    sanity_check(config)
    return config


NUM_RETRY = 5

def generate_indices_tuples(n, n_tuples, k=3, offset=0):
    random.seed(42)
    unique_sets = set()
    result = []
    n_list = None
    if isinstance(n, list):
        n_list = n
    else:
        n_list = [x + offset for x in list(range(n))]

    if len(n_list) < k:
        k = len(n_list)

    retry = NUM_RETRY
    while len(result) < n_tuples and retry > 0:
        sample = tuple(sorted(random.sample(n_list, k)))
        if sample not in unique_sets:
            unique_sets.add(sample)
            result.append(list(sample))
            retry = NUM_RETRY
        else:
            retry -= 1

    return result

