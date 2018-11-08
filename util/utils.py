import json
import logging

logging.basicConfig(level=logging.INFO, format='')


def get_instance(module, name, config, *args):
    func_args = config[name]['args'] if 'args' in config[name] else None

    if func_args:
        return getattr(module, config[name]['type'])(*args, **func_args)
    else:
        return getattr(module, config[name]['type'])(*args)


def save_json(x, fname):
    with open(fname, 'w') as outfile:
        json.dump(x, outfile)


class Logger:
    """
    Training process logger
    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
