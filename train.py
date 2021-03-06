import os
import json
import random
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_model
from trainer import Trainer
from util import Logger, get_instance


def main(config, resume):
    random.seed(0)
    torch.manual_seed(0)
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    list_compose = {i: get_instance(module_model, i, config) for i in config['model_compose']}
    config['arch']['args'] = list_compose
    model = get_instance(module_model, 'arch', config)
    model.summary()
    config['arch'].pop('args', None)

    # get function handles of loss and metrics
    loss = {config[i]['type']: get_instance(module_loss, i, config) for i in config if 'loss' in i}

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    #trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, model.parameters())

    trainer = Trainer(model, loss, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
