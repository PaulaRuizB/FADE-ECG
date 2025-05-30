import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append('../')
from utils.loss_functions import mse_inside_outside_thresholds
from torch.utils.data import DataLoader
from timm import utils

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            save_path_train: str,
            loss1: str,
            batch_size: int,
            writer,
            lr_scheduler,
            distributed_args,
            amp_autocast,
            ft,
            num_channels,
            val_data: DataLoader,
            weights_mse_inorout_lastsecond,
            threshold_mse_inside_outside,
            path_train_set,
            norm_a_b_max_min,
            dataset_FT,
            path_datasetinfo,
            input_size_seconds,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        if distributed_args.distributed:
            self.model = DDP(model, device_ids=[gpu_id])
        self.distributed_args = distributed_args
        self.save_path_train = save_path_train
        self.loss1 = loss1
        self.batch_size = batch_size
        self.writer = writer
        self.lr_scheduler = lr_scheduler
        self.amp_autocast = amp_autocast
        self.weights_mse_inorout_lastsecond = weights_mse_inorout_lastsecond
        self.threshold_mse_inside_outside = threshold_mse_inside_outside
        self.ft = ft
        self.num_channels = num_channels
        self.path_train_set = path_train_set
        self.norm_a_b_max_min = norm_a_b_max_min
        self.dataset_FT = dataset_FT
        self.path_datasetinfo = path_datasetinfo
        self.input_size_seconds = input_size_seconds

    def _run_batch(self, source, targets, num_updates, epoch):

        self.optimizer.zero_grad()
        self.model.train()

        source = source.half()
        targets = targets.half()

        with self.amp_autocast():

            if self.loss1 == "mse_inside_outside_thresholds":
                output = self.model(source)
                loss_1, loss_2 = mse_inside_outside_thresholds(targets, output, self.threshold_mse_inside_outside[1], self.threshold_mse_inside_outside[0], "mean")
                loss_1 = loss_1 * self.weights_mse_inorout_lastsecond[0]
                loss_2 = loss_2 * self.weights_mse_inorout_lastsecond[1]
                loss = loss_1 + loss_2
            else:
                output = self.model(source)
                loss = F.mse_loss(output, targets, reduction="mean")

        loss.backward()
        self.optimizer.step()

        lrl = [param_group['lr'] for param_group in self.lr_scheduler.optimizer.param_groups]
        lr_value = sum(lrl) / len(lrl)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step_update(num_updates=num_updates, metric=loss)

        if self.loss1 == 'mse_inside_outside_thresholds':
            return loss, output, lr_value, loss_1, loss_2
        else:
            return loss, output, lr_value

    def _run_batch_val(self, source, targets):

        source = source.half()
        targets = targets.half()

        self.model.eval()

        with torch.no_grad():

            with self.amp_autocast():

                if self.loss1 == "mse_inside_outside_thresholds":
                    output = self.model(source)
                    loss_1, loss_2 = mse_inside_outside_thresholds(targets, output,
                                                                   self.threshold_mse_inside_outside[1],
                                                                   self.threshold_mse_inside_outside[0], "mean")
                    loss_1 = loss_1 * self.weights_mse_inorout_lastsecond[0]
                    loss_2 = loss_2 * self.weights_mse_inorout_lastsecond[1]
                    loss = loss_1 + loss_2
                else:
                    output = self.model(source)
                    loss = F.mse_loss(output, targets, reduction="mean")

        if self.loss1 == 'mse_inside_outside_thresholds':
            return loss, output, loss_1, loss_2
        else:
            return loss, output


    def _run_epoch(self, epoch, distributed_args):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if distributed_args.distributed:
            self.train_data.sampler.set_epoch(epoch)
            self.val_data.sampler.set_epoch(epoch)
        else:
            print("Nothing to do, only one GPU")

        accum_steps = 1
        updates_per_epoch = (len(self.train_data) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch

        for batch_idx, (source, targets) in enumerate(self.train_data):

            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)

            if source.shape[0] == self.batch_size:
                if self.loss1 == 'mse_inside_outside_thresholds':
                    value_loss, output, lr_value, loss_1, loss_2 = self._run_batch(source, targets, num_updates, epoch)
                else:
                    value_loss, output, lr_value = self._run_batch(source, targets, num_updates, epoch)

            if utils.is_primary(distributed_args) and (batch_idx == len(self.train_data)-1):

                fig1 = plt.figure()
                for i in range(self.num_channels):
                    if i==0 and self.num_channels>1:
                        plt.plot(source[0][0].cpu()*10.0)
                    else:
                        plt.plot(source[0][i].cpu())
                self.writer.add_figure('Input', fig1, len(self.train_data)*epoch + batch_idx)

                fig2 = plt.figure()
                for i in range(self.num_channels):
                    if i==0 and self.num_channels>1:
                        plt.plot(targets[0][0].cpu()*10.0)
                    else:
                        plt.plot(targets[0][i].cpu())
                self.writer.add_figure('Label', fig2, len(self.train_data) * epoch + batch_idx)

                fig3 = plt.figure()
                for i in range(self.num_channels):
                    if i==0 and self.num_channels>1:
                        plt.plot(output[0][0].cpu().detach().numpy()*10.0)
                    else:
                        plt.plot(output[0][i].cpu().detach().numpy())
                self.writer.add_figure('Output', fig3, len(self.train_data) * epoch + batch_idx)


            if utils.is_primary(distributed_args) and (batch_idx == len(self.train_data)-1):
                print(f"Train: {epoch} [{batch_idx:>4d}/{len(self.train_data)}], loss: {value_loss}, learning rate: {lr_value}")
                self.writer.add_scalar("Loss_steps", value_loss, len(self.train_data)*epoch + batch_idx)


        for batch_idx, (source_val, targets_val) in enumerate(self.val_data):

            source_val = source_val.to(self.gpu_id)
            targets_val = targets_val.to(self.gpu_id)

            if self.loss1 == 'mse_inside_outside_thresholds':
                value_loss_val, output_val, loss_1_val, loss_2_val = self._run_batch_val(source_val, targets_val)
            else:
                value_loss_val, output_val = self._run_batch_val(source_val, targets_val)


            if utils.is_primary(distributed_args) and (batch_idx == len(self.val_data)-1):

                fig1 = plt.figure()
                for i in range(self.num_channels):
                    if i==0  and self.num_channels>1:
                        plt.plot(source_val[0][0].cpu()*10.0)
                    else:
                        plt.plot(source_val[0][i].cpu())
                self.writer.add_figure('Input_val', fig1, len(self.val_data)*epoch + batch_idx)

                fig2 = plt.figure()
                for i in range(self.num_channels):
                    if i==0 and self.num_channels>1:
                        plt.plot(targets_val[0][0].cpu()*10.0)
                    else:
                        plt.plot(targets_val[0][i].cpu())
                self.writer.add_figure('Label_val', fig2, len(self.val_data) * epoch + batch_idx)

                fig3 = plt.figure()
                for i in range(self.num_channels):
                    if i==0 and self.num_channels>1:
                        plt.plot(output_val[0][0].cpu().detach().numpy()*10.0)
                    else:
                        plt.plot(output_val[0][i].cpu().detach().numpy())
                self.writer.add_figure('Output_val', fig3, len(self.val_data) * epoch + batch_idx)


            if utils.is_primary(distributed_args) and (batch_idx == len(self.val_data)-1):
                print(f"Validation: {epoch} [{batch_idx:>4d}/{len(self.val_data)}], loss: {value_loss_val}, learning rate: {lr_value}")
                self.writer.add_scalar("Loss_steps_val", value_loss_val, len(self.val_data)*epoch + batch_idx)

        if self.loss1 == 'mse_inside_outside_thresholds':
            return value_loss, lr_value,value_loss_val, loss_1, loss_2, loss_1_val, loss_2_val
        else:
            return value_loss, lr_value, value_loss_val

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        save_path = self.save_path_train + '/checkpoint_{}.pt'.format(epoch)
        torch.save(ckp, save_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            if self.loss1 == 'mse_inside_outside_thresholds':
                value_loss, lr_value, value_loss_val, loss_1, loss_2, loss_1_val, loss_2_val  = self._run_epoch(epoch, self.distributed_args)
            else:
                value_loss, lr_value, value_loss_val = self._run_epoch(epoch, self.distributed_args)

            if utils.is_primary(self.distributed_args):
                self.writer.add_scalar("Loss/train", value_loss, epoch)
                print("Loss value {} in epoch {}".format(value_loss, epoch))
                self.writer.add_scalar("lr", lr_value, epoch)
                if self.loss1 == 'mse_inside_outside_thresholds':
                    self.writer.add_scalar("Loss_train_combined/loss_1", loss_1, epoch)
                    self.writer.add_scalar("Loss_train_combined/loss_2", loss_2, epoch)

                self.writer.add_scalar("Loss/val", value_loss_val, epoch)
                print("Loss value val {} in epoch {}".format(value_loss_val, epoch))
                if self.loss1 == 'mse_inside_outside_thresholds':
                    self.writer.add_scalar("Loss_val_combined/loss_1", loss_1_val, epoch)
                    self.writer.add_scalar("Loss_val_combined/loss_2", loss_2_val, epoch)

            if utils.is_primary(self.distributed_args) and (((epoch+1) % self.save_every) == 0):
                self._save_checkpoint(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch+1)

        print("Train DONE!!!")

        self.writer.close()
