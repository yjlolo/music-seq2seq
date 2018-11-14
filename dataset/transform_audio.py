import os
import json
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torchvision import transforms
import dataset.datasets as module_dataset
import dataset.transformers as module_transform
from util import get_instance, save_json, ensure_dir


def main(config):
    list_transform = [get_instance(module_transform, i, config) for i in config if 'transform' in i]
    transform = transforms.Compose(list_transform)
    config['dataset']['args']['transform'] = transform

    d = get_instance(module_dataset, 'dataset', config)

    save_path = os.path.join(config['save_dir'], config['save_subdir'])

    ensure_dir(save_path)

    config['dataset']['args'].pop('transform', None)
    save_json(config, os.path.join(save_path, 'transform_config.json'))

    np.random.seed(0)
    samples_for_draw = np.random.choice(len(d), size=9, replace=False)
    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(15, 15))
    n_fig = 0

    start_time = time.time()
    for k in range(len(d)):
        print("Transforming %d-th data ..." % k)
        data = d[k]
        x, _, songid = data[0], data[1], data[2]

        if isinstance(x, list):
            for i, x_i in enumerate(x):
                p = os.path.join(save_path, '%s-%s.%s' % (songid, i, config['save_subdir']))
                torch.save(x_i, p)
            x = np.vstack(x)
        else:
            p = os.path.join(save_path, '%s.%s' % (songid, config['save_subdir']))
            torch.save(x, p)

        if k in samples_for_draw:
            ax = fig.add_subplot(gs[n_fig])
            if isinstance(x, torch.Tensor):
                ax.imshow(x.cpu().data.numpy(), aspect='auto', origin='lower')
            else:
                ax.imshow(x, aspect='auto', origin='lower')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            n_fig += 1

    plt.savefig(os.path.join(save_path, '.'.join(['spec', 'jpg'])))

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
