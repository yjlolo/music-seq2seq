import numpy as np
import time
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        epoch_start_time = time.time()
        total_loss = 0
        for batch_idx, batch_d in enumerate(self.data_loader):
            # refer to data_loader/collates.py to see the five returned items
            x, y, seqlen, songid, mask = batch_d[0], batch_d[1], batch_d[2], batch_d[3], batch_d[4]
            batch_size = x.size(0)
            input_size = x.size(-1)

            input_var = x.to(self.device)
            centroids = y.type(input_var.type())  # used for loss constraints
            mask = mask.to(self.device)
            eff_len = torch.FloatTensor(seqlen).sum().to(self.device)  # used for loss normalization

            sos = torch.zeros(batch_size, 1, input_size).to(self.device)  # start-of-sequence dummy
            target_var = torch.cat((sos, input_var), dim=1)  # only use teacher forcing currently

            # model update
            self.optimizer.zero_grad()
            output, enc_outputs = self.model(input_var, target_var, seqlen)
            loss = self.loss(output.contiguous().view(-1), input_var.view(-1)).mul(mask.view(-1)).sum()  # mask padded
            loss = loss.div(eff_len).div(input_size)  # loss normalization to per frequency bin
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'time': time.time() - epoch_start_time
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        epoch_start_time = time.time()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch_d in enumerate(self.valid_data_loader):
                x, y, seqlen, songid, mask = batch_d[0], batch_d[1], batch_d[2], batch_d[3], batch_d[4]
                batch_size = x.size(0)
                input_size = x.size(-1)

                input_var = x.to(self.device)
                centroids = y.type(input_var.type())  # used for loss constraints
                mask = mask.to(self.device)
                eff_len = torch.FloatTensor(seqlen).sum().to(self.device)  # used for loss normalization

                sos = torch.zeros(batch_size, 1, input_size).to(self.device)  # start-of-sequence dummy
                target_var = torch.cat((sos, input_var), dim=1)  # only use teacher forcing currently

                output, enc_outputs = self.model(input_var, target_var, seqlen)
                loss = self.loss(output.contiguous().view(-1), input_var.view(-1)).mul(mask.view(-1)).sum()
                loss = loss.div(eff_len).div(input_size)  # loss normalization to per frequency bin

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_time': time.time() - epoch_start_time
        }
