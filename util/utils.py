import json


def get_instance(module, name, config, *args):
    func_args = config[name]['args'] if 'args' in config[name] else None

    if func_args:
        return getattr(module, config[name]['type'])(*args, **func_args)
    else:
        return getattr(module, config[name]['type'])(*args)


def save_json(x, fname):
    with open(fname, 'w') as outfile:
        json.dump(x, outfile)
