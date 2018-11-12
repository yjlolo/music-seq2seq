import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math
import torch
import data_loader.data_loaders as module_data
import model.model as module_model
from util import get_instance, ensure_dir


def main(config, resume, figsave=None):
    torch.manual_seed(0)

    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader#.split_validation()

    list_compose = {i: get_instance(module_model, i, config) for i in config['model_compose']}
    config['arch']['args'] = list_compose
    model = get_instance(module_model, 'arch', config)
    model.summary()

    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X = forwardpass(valid_data_loader, model, device, figsave)

    if figsave:
        target_resume = resume.split('/')[-3:-1]
        project_name, date = target_resume[0], target_resume[1]
        fig_dir = os.path.join(args.figure, project_name)
        ensure_dir(fig_dir)
        plt.savefig(os.path.join(fig_dir, '.'.join([date, 'jpg'])))

    return X


def draw(fig, gs, k, x):
    ax = fig.add_subplot(gs[k])
    ax.imshow(x, aspect='auto', origin='lower')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def forwardpass(data_loader, model, device, figsave):
    if figsave:
        # N = data_loader.batch_size * len(data_loader)
        n_sample = 4  # number of samples for visualization
        N = n_sample * 2  # each is a original-reconstructed pair
        cols = n_sample
        rows = int(math.ceil(N / cols))
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure(figsize=(20, 10))

        # sample
        np.random.seed(0)
        song_ids = np.array(list(map(lambda x: x[0].split('/')[-1].split('.')[0], data_loader.dataset.data)))
        songids = song_ids[data_loader.sampler.indices]
        fig_sample = np.random.choice(songids, size=n_sample, replace=False)
        fig_sample

    with torch.no_grad():
        input_feat = []
        k = 0
        for batch_idx, batch_d in enumerate(data_loader):
            x, y, seqlen, songid, mask = batch_d[0], batch_d[1], batch_d[2], batch_d[3], batch_d[4]
            batch_size = x.size(0)
            input_size = x.size(-1)
            original_input = x.to(device)

            x_np = np.flip(x.numpy(), 0).copy()  # Reverse of copy of numpy array of given tensor
            x_r = torch.from_numpy(x_np).to(device)

            input_var = x_r.to(device)
            centroids = y.type(input_var.type())  # used for loss constraints
            mask = mask.to(device)
            eff_len = torch.FloatTensor(seqlen).sum().to(device)  # used for loss normalization

            sos = torch.zeros(batch_size, 1, input_size).to(device)  # start-of-sequence dummy
            target_var = torch.cat((sos, original_input), dim=1)  # only use teacher forcing currently

            output, enc_outputs = model(input_var, target_var, input_lengths=seqlen, train=False)
            input_feat.append(enc_outputs.view(batch_size, -1))

            if figsave:
                for i, (x_ori, x_hat, sid) in enumerate(zip(original_input.cpu().data.numpy(),
                                                            output.cpu().data.numpy(),
                                                            songid.data.numpy()
                                                            )):
                    if str(sid) in fig_sample:
                        draw(fig, gs, k, x_hat.T)
                        draw(fig, gs, k + n_sample, x_ori.T)
                        k += 1

    return torch.cat(input_feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--figure', default=None, type=str,
                        help='path to save reconstructed spectrogram (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICERS"] = args.device

    y = main(config, args.resume, args.figure)
