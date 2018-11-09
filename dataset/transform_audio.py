import os
import json
import time
import argparse
import torch
from torchvision import transforms
import dataset.datasets as module_dataset
import dataset.transformers as module_transform
from util import get_instance, save_json, ensure_dir


def main(config):
    list_transform = [i for i in config if 'transform' in i]
    list_transform = [get_instance(module_transform, j, config)
                      for j in list_transform]
    transform = transforms.Compose(list_transform)
    config['dataset']['args']['transform'] = transform

    d = get_instance(module_dataset, 'dataset', config)

    save_path = os.path.join(config['save_dir'], config['save_subdir'])

    ensure_dir(save_path)

    config['dataset']['args'].pop('transform', None)
    save_json(config, os.path.join(save_path, 'transform_config.json'))

    start_time = time.time()
    for k in range(len(d)):
        print("Transforming %d-th data ..." % k)
        data = d[k]
        x, _, songid = data[0], data[1], data[2]
        p = os.path.join(save_path, '%s.%s' % (songid, config['save_subdir']))

        torch.save(x, p)

    print("Time: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio transformer')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified.")

    d = main(config)
