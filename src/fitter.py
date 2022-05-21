# Requirements
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Iterable, Callable, Tuple
from datetime import datetime
import time
import os


# AverageMeter class (taken from benatools: https://github.com/benayas1/benatools)
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Attributes
    ----------
    val : float
        Stores the average loss of the last batch
    avg : float
        Average loss
    sum : float
        Sum of all losses
    count : int
        number of elements
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates current internal state
        Parameters
        ----------
        val : float
            loss on each training step
        n : int, Optional
            batch size
        """
        if np.isnan(val) or np.isinf(val):
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Fitter base class (taken from benatools: https://github.com/benayas1/benatools)
class TorchFitterBase:
    """
    Helper class to implement a training loop in PyTorch
    """

    def __init__(self,
                 model: torch.nn.Module = None,
                 device: str = 'cpu',
                 loss: torch.nn.Module = None,
                 optimizer: torch.optim = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 validation_scheduler: bool = True,
                 step_scheduler: bool = False,
                 folder: str = 'models',
                 verbose: bool = True,
                 save_log: bool = True,
                 use_amp: bool = False,
                 ):
        """
        Args:
            model (torch.nn.Module): Model to be fitted
            device (str): Device can be cuda or cpu
            loss (torch.nn.Module): DataFrame to split
            optimizer (torch.optim): Optimizer object
            scheduler (torch.optim.lr_scheduler, optional): Scheduler object. Defaults to None.
            validation_scheduler (bool, optional): Run scheduler step on the validation step. Defaults to True.
            step_scheduler (bool, optional): Run scheduler step on every training step. Defaults to False.
            folder (str, optional): Folder where to store checkpoints. Defaults to 'models'.
            verbose (bool, optional): Whether to print outputs or not. Defaults to True.
            save_log (bool, optional): Whether to write the log in log.txt or not. Defaults to True.
        """
        if loss is not None:
            if type(loss) == type:
                self.loss_function = loss()
            else:
                self.loss_function = loss
        else:
            self.loss_function = None

        self.epoch = 0  # current epoch
        self.verbose = verbose

        self.base_dir = f'{folder}'

        self.save_log = save_log
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_metric = 0

        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Optimizer object
        self.optimizer = optimizer

        # Scheduler Object
        self.scheduler = scheduler
        self.validation_scheduler = validation_scheduler  # do scheduler.step after validation stage loss
        self.step_scheduler = step_scheduler  # do scheduler.step after optimizer.step
        self.log(f'Fitter prepared. Device is {self.device}')

    def unpack(self, data):
        raise NotImplementedError('This class is a base class')

    def reduce_loss(self, loss, weights):
        # Apply sample weights if existing
        if len(loss.shape) > 0:
            # apply weights
            if weights is not None:
                loss = loss * torch.unsqueeze(weights, 1)

            # reduction
            loss = loss.mean()
        return loss

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            n_epochs: int = 1,
            metrics: Iterable[Tuple[Callable[[Iterable, Iterable], float], dict]] = None,
            early_stopping: int = 0,
            early_stopping_mode: str = 'min',
            early_stopping_alpha: float = 0.0,
            early_stopping_pct: float = 0.0,
            save_checkpoint: bool = False,
            save_best_checkpoint: bool = True,
            verbose_steps: int = 0,
            callbacks: Iterable[Callable[[Dict], None]] = None):
        """
        Fits a model
        Args:
            train_loader (torch.utils.data.DataLoader): Training data
            val_loader (torch.utils.data.DataLoader, optional): Validation Data. Defaults to None.
            n_epochs (int, optional): Maximum number of epochs to train. Defaults to 1.
            metrics ( function with (y_true, y_pred, **metric_kwargs) signature, optional): Metric to evaluate results on. Defaults to None.
            metric_kwargs (dict, optional): Arguments for the passed metric. Ignored if metric is None. Defaults to {}.
            early_stopping (int, optional): Early stopping epochs. Defaults to 0.
            early_stopping_mode (str, optional): Min or max criteria. Defaults to 'min'.
            early_stopping_alpha (float, optional): Value that indicates how much to improve to consider early stopping. Defaults to 0.0.
            early_stopping_pct (float, optional): Value between 0 and 1 that indicates how much to improve to consider early stopping. Defaults to 0.0.
            save_checkpoint (bool, optional): Whether to save the checkpoint when training. Defaults to False.
            save_best_checkpoint (bool, optional): Whether to save the best checkpoint when training. Defaults to True.
            verbose_steps (int, optional): Number of step to print every training summary. Defaults to 0.
            callbacks (list of callable, optional): List of callback functions to be called after an epoch
        Returns:
            pd.DataFrame: DataFrame containing training history
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        if self.best_metric == 0.0:
            self.best_metric = np.inf if early_stopping_mode == 'min' else -np.inf

        initial_epochs = self.epoch

        # Use the same train loader for validation. A possible use case is for autoencoders
        if isinstance(val_loader, str) and val_loader == 'training':
            val_loader = train_loader

        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.utcnow().isoformat(" ", timespec="seconds")}\n \
                        EPOCH {str(self.epoch+1)}/{str(n_epochs+initial_epochs)} - LR: {lr}')

            # Run one training epoch
            t = time.time()
            train_summary_loss = self.train_one_epoch(train_loader, verbose_steps=verbose_steps)
            history['train'] = train_summary_loss.avg  # training loss
            history['lr'] = self.optimizer.param_groups[0]['lr']

            # Save checkpoint
            if save_checkpoint:
                self.save(f'{self.base_dir}/last-checkpoint.bin', False)

            if val_loader is not None:
                # Run epoch validation
                val_summary_loss, calculated_metrics = self.validation(val_loader,
                                                                       metric=metrics,
                                                                       verbose_steps=verbose_steps)
                history['val'] = val_summary_loss.avg  # validation loss

                # Write log
                metric_log = ' - ' + ' - '.join([f'{fname}: {value}' for value, fname in calculated_metrics]) if calculated_metrics else ''
                self.log(f'\r[RESULT] {(time.time() - t):.2f}s - train loss: {train_summary_loss.avg:.5f} - val loss: {val_summary_loss.avg:.5f}' + metric_log)

                if calculated_metrics:
                    history.update({fname: value for value, fname in calculated_metrics})
                    #history['val_metric'] = calculated_metrics

                calculated_metric = calculated_metrics[0][0] if calculated_metrics else val_summary_loss.avg
            else:
                # If no validation is provided, training loss is used as metric
                calculated_metric = train_summary_loss.avg

            es_pct = early_stopping_pct * self.best_metric

            # Check if result is improved, then save model
            if (
                ((metrics) and
                 (
                  ((early_stopping_mode == 'max') and (calculated_metric - max(early_stopping_alpha, es_pct) > self.best_metric)) or
                  ((early_stopping_mode == 'min') and (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric))
                 )
                ) or
                ((metrics is None) and
                 (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric) # the standard case is to minimize
                )
               ):
                self.log(f'Validation metric improved from {self.best_metric} to {calculated_metric}')
                self.best_metric = calculated_metric
                self.model.eval()
                if save_best_checkpoint:
                    savepath = f'{self.base_dir}/best-checkpoint.bin'
                    self.save(savepath)
                es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Callbacks receive the history dict of this epoch
            if callbacks is not None:
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                for c in callbacks:
                    c(history)

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs >= early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            # Scheduler step after validation
            if self.validation_scheduler and self.scheduler is not None:
                self.scheduler.step(metrics=calculated_metric)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')

    def train_one_epoch(self, train_loader, verbose_steps=0):
        """
        Run one epoch on the train dataset
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            DataLoaders containing the training dataset
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()
        batch_size = train_loader.batch_size

        # run epoch
        for step, data in enumerate(train_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_loader)} | ' +
                        f'summary_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs | ' +
                        f'ETA: {(len(train_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            # Unpack batch of data
            x, y, w = self.unpack(data)

            # Run one batch
            loss = self.train_one_batch(x, y, w)

            summary_loss.update(loss.detach().item(), batch_size)

            # update optimizer using mixed precision if requested
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # LR Scheduler step after epoch
            if self.step_scheduler and self.scheduler is not None:
                self.scheduler.step()

        self.log(f'\r[TRAIN] {(time.time() - t):.2f}s - train loss: {summary_loss.avg:.5f}')

        return summary_loss

    def train_one_batch(self, x, y, w=None):
        """
        Trains one batch of data.
        The actions to be done here are:
        - extract x and y (labels)
        - calculate output and loss
        - backpropagate
        Args:
            x (List or Tuple or Dict): Data
            y (torch.Tensor): Labels
            w (torch.Tensor, optional): Weights. Defaults to None.
        Returns:
            torch.Tensor: A tensor with the calculated loss
        """
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Output and loss
            if isinstance(x, tuple) or isinstance(x, list):
                output = self.model(*x)
            elif isinstance(x, dict):
                output = self.model(**x)
            else:
                output = self.model(x)

            loss = self.loss_function(output, y)

            # Reduce loss and apply sample weights if existing
            loss = self.reduce_loss(loss, w)
        
        # backpropagation
        self.scaler.scale(loss).backward()


        return loss

    def validation(self, val_loader, metric=None, verbose_steps=0):
        """
        Validates a model
        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation Data
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        float
            Calculated metric if a metric is provided, else None
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        self.model.eval()
        summary_loss = AverageMeter()
        y_preds = []
        y_true = []
        batch_size = val_loader.batch_size

        t = time.time()
        for step, data in enumerate(val_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rVal Step {step}/{len(val_loader)} | ' +
                        f'summary_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(val_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x, y, w = self.unpack(data)

                if metric:
                    y_true += y.cpu().numpy().tolist()

                # just forward pass
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)

                loss = self.loss_function(output, y)

                # Reduce loss and apply sample weights if existing
                loss = self.reduce_loss(loss, w)
                summary_loss.update(loss.detach().item(), batch_size)

                if metric:
                    y_preds += output.cpu().numpy().tolist()

        # Callback metrics
        metric_log = ' '*30
        if metric:
            calculated_metrics = []
            y_pred = np.argmax(y_preds, axis=1)
            for f, args in metric:
                value = f(y_true, y_pred, **args)
                calculated_metrics.append((value, f.__name__))
                metric_log = f'- {f.__name__} {value:.5f} '
        else:
            calculated_metrics = None

        self.log(f'\r[VALIDATION] {(time.time() - t):.2f}s - val. loss: {summary_loss.avg:.5f} ' + metric_log)
        return summary_loss, calculated_metrics

    def predict(self, test_loader, verbose_steps=0):
        """
        Makes predictions using the trained model
        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test Data
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        np.array
            Predicted values by the model
        """
        if self.model is None:
            self.log(f"ERROR: Model is not existing.")
            raise ValueError(f"ERROR: Model is not existing.")

        self.model.eval()
        y_preds = []
        t = time.time()

        for step, data in enumerate(test_loader):
            if self.verbose & (verbose_steps > 0) > 0:
                if step % verbose_steps == 0:
                    print(
                        f'\rPrediction Step {step}/{len(test_loader)} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(test_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x, _, _ = self.unpack(data)

                # Output
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)

                y_preds += output.cpu().numpy().tolist()

        return np.array(y_preds)

    def save(self, path, verbose=True):
        """
        Save model and other metadata
        Args:
            path (str): Path of the file to be saved
            verbose (bool, optional): True = print logs, False = silence. Defaults to True.
        """

        if verbose:
            self.log(f'Checkpoint is saved to {path}')
        self.model.eval()

        data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_summary_loss': self.best_metric,
                'epoch': self.epoch,
                'scaler': self.scaler.state_dict()
        }

        if self.scheduler is not None:
            data['scheduler_state_dict'] = self.scheduler.state_dict()

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        torch.save(data, path)

    def load(self, path, only_model=False):
        """
        Load model and other metadata
        Args:
            path (str): Path of the file to be loaded
            only_model (bool, optional): Whether to load just the model weights. Defaults to False.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if only_model:
            return

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_metric = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @staticmethod
    def load_model_weights(path, model):
        """
        Static method that loads weights into a torch module, extracted from a checkpoint
        Args:
            path (str): Path containing the weights. Normally a .bin or .tar file
            model (torch.nn.Module): Module to load the weights on
        Returns:
            torch.nn.Module: The input model with loaded weights
        """

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def log(self, message):
        """
        Log training ouput into console and file
        Args:
            message (str): Message to be logged
        """
        if self.verbose:
            print(message)

        if self.save_log is True:
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{message}\n')



# Fitter method
class IC_NER_Fitter(TorchFitterBase):

    # Mechanism to unwrap data and place it in the right device
    def unpack(self, data):
        x = {k: v.to(self.device) for k, v in data[0].items()}
        
        y = {k: v.to(self.device) for k, v in data[1].items()}

        if 'w' in data:
            w = data['w'].to(self.device).float()
        else:
            w = None

        return x, y, w

    # Fit
    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            n_epochs: int = 1,
            metrics: Iterable[Tuple[Callable[[Iterable, Iterable], float], dict]] = None,
            early_stopping: int = 0,
            early_stopping_mode: str = 'min',
            early_stopping_alpha: float = 0.0,
            early_stopping_pct: float = 0.0,
            save_checkpoint: bool = False,
            save_best_checkpoint: bool = True,
            verbose_steps: int = 0,
            callbacks: Iterable[Callable[[Dict], None]] = None):
        """
        Fits a model
        Args:
            train_loader (torch.utils.data.DataLoader): Training data
            val_loader (torch.utils.data.DataLoader, optional): Validation Data. Defaults to None.
            n_epochs (int, optional): Maximum number of epochs to train. Defaults to 1.
            metrics ( function with (y_true, y_pred, **metric_kwargs) signature, optional): Metric to evaluate results on. Defaults to None.
            metric_kwargs (dict, optional): Arguments for the passed metric. Ignored if metric is None. Defaults to {}.
            early_stopping (int, optional): Early stopping epochs. Defaults to 0.
            early_stopping_mode (str, optional): Min or max criteria. Defaults to 'min'.
            early_stopping_alpha (float, optional): Value that indicates how much to improve to consider early stopping. Defaults to 0.0.
            early_stopping_pct (float, optional): Value between 0 and 1 that indicates how much to improve to consider early stopping. Defaults to 0.0.
            save_checkpoint (bool, optional): Whether to save the checkpoint when training. Defaults to False.
            save_best_checkpoint (bool, optional): Whether to save the best checkpoint when training. Defaults to True.
            verbose_steps (int, optional): Number of step to print every training summary. Defaults to 0.
            callbacks (list of callable, optional): List of callback functions to be called after an epoch
        Returns:
            pd.DataFrame: DataFrame containing training history
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        if self.best_metric == 0.0:
            self.best_metric = np.inf if early_stopping_mode == 'min' else -np.inf

        initial_epochs = self.epoch

        # Use the same train loader for validation. A possible use case is for autoencoders
        if isinstance(val_loader, str) and val_loader == 'training':
            val_loader = train_loader

        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.utcnow().isoformat(" ", timespec="seconds")}\n \
                        EPOCH {str(self.epoch+1)}/{str(n_epochs+initial_epochs)} - LR: {lr}')

            # Run one training epoch
            t = time.time()
            train_loss = self.train_one_epoch(train_loader, verbose_steps=verbose_steps)
            history['train_ic_loss'] = train_loss['IC'].avg
            history['train_ner_loss'] = train_loss['NER'].avg
            history['train_summary_loss'] = train_loss['summary'].avg
            history['lr'] = self.optimizer.param_groups[0]['lr']

            # Save checkpoint
            if save_checkpoint:
                self.save(f'{self.base_dir}/last-checkpoint.bin', False)

            if val_loader is not None:
                # Run epoch validation
                val_loss, calculated_metrics = self.validation(val_loader,
                                                               metric=metrics,
                                                               verbose_steps=verbose_steps)
                
                history['val_ic_loss'] = val_loss['IC'].avg
                history['val_ner_loss'] = val_loss['NER'].avg
                history['val_summary_loss'] = val_loss['summary'].avg

                # Write log
                metric_log = ' - ' + ' - '.join([f'{fname}: {value}' for value, fname in calculated_metrics]) if calculated_metrics else ''
                self.log(f"\r[RESULT] {(time.time() - t):.2f}s - "\
                         f"train ic loss: {train_loss['IC'].avg:.5f} - "\
                         f"train ner loss: {train_loss['NER'].avg:.5f} - "\
                         f"train summary loss: {train_loss['summary'].avg:.5f} - "\
                         f"val ic loss: {val_loss['IC'].avg:.5f} - "\
                         f"val ner loss: {val_loss['NER'].avg:.5f} - "\
                         f"val summary loss: {val_loss['summary'].avg:.5f} - "\
                         + metric_log
                         )

                if calculated_metrics:
                    history.update({fname: value for value, fname in calculated_metrics})
                    #history['val_metric'] = calculated_metrics

                calculated_metric = calculated_metrics[0][0] if calculated_metrics else val_loss['summary'].avg
            else:
                # If no validation is provided, training loss is used as metric
                calculated_metric = train_loss['summary'].avg

            es_pct = early_stopping_pct * self.best_metric

            # Check if result is improved, then save model
            if (
                ((metrics) and
                 (
                  ((early_stopping_mode == 'max') and (calculated_metric - max(early_stopping_alpha, es_pct) > self.best_metric)) or
                  ((early_stopping_mode == 'min') and (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric))
                 )
                ) or
                ((metrics is None) and
                 (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric) # the standard case is to minimize
                )
               ):
                self.log(f'Validation metric improved from {self.best_metric} to {calculated_metric}')
                self.best_metric = calculated_metric
                self.model.eval()
                if save_best_checkpoint:
                    savepath = f'{self.base_dir}/best-checkpoint.bin'
                    self.save(savepath)
                es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Callbacks receive the history dict of this epoch
            if callbacks is not None:
                if not isinstance(callbacks, list):
                    callbacks = [callbacks]
                for c in callbacks:
                    c(history)

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs >= early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            # Scheduler step after validation
            if self.validation_scheduler and self.scheduler is not None:
                self.scheduler.step(metrics=calculated_metric)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')

    # Epoch
    def train_one_epoch(self, train_loader, verbose_steps=0):
        """
        Run one epoch on the train dataset
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            DataLoaders containing the training dataset
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_ic_loss = AverageMeter()  # object to track the average loss
        summary_ner_loss = AverageMeter()  # object to track the average loss
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()
        batch_size = train_loader.batch_size

        # run epoch
        for step, data in enumerate(train_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_loader)} | ' +
                        f'ic_loss: {summary_ic_loss.avg:.5f} | ' +
                        f'ner_loss: {summary_ner_loss.avg:.5f} | ' +
                        f'summary_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs | ' +
                        f'ETA: {(len(train_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            # Unpack batch of data
            x, y, w = self.unpack(data)

            # Run one batch
            loss = self.train_one_batch(x, y, w)

            summary_ic_loss.update(loss['IC'].detach().item(), batch_size)
            summary_ner_loss.update(loss['NER'].detach().item(), batch_size)
            summary_loss.update(loss['summary'].detach().item(), batch_size)

            # update optimizer using mixed precision if requested
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # LR Scheduler step after epoch
            if self.step_scheduler and self.scheduler is not None:
                self.scheduler.step()

        self.log(f"\r[TRAIN] {(time.time() - t):.2f}s - "\
                 f"train ic loss: {summary_ic_loss.avg:.5f} - "\
                 f"train ner loss: {summary_ner_loss.avg:.5f} - "\
                 f"train summary loss: {summary_loss.avg:.5f} - "\
                 )

        return {'IC':summary_ic_loss, 'NER':summary_ner_loss, 'summary':summary_loss}

    def train_one_batch(self, x, y, w=None):
        """
        Trains one batch of data.
        The actions to be done here are:
        - extract x and y (labels)
        - calculate output and loss
        - backpropagate
        Args:
            x (List or Tuple or Dict): Data
            y (torch.Tensor): Labels
            w (torch.Tensor, optional): Weights. Defaults to None.
        Returns:
            torch.Tensor: A tensor with the calculated loss
        """
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Output and loss
            if isinstance(x, tuple) or isinstance(x, list):
                output = self.model(*x)
            elif isinstance(x, dict):
                output = self.model(**x)
            else:
                output = self.model(x)

            loss = self.loss_function(output, y)

            # Reduce loss and apply sample weights if existing
            loss['summary'] = self.reduce_loss(loss['summary'], w)
        
        # backpropagation
        self.scaler.scale(loss['summary']).backward()


        return loss

    #Validation
    def validation(self, val_loader, metric=None, verbose_steps=0):
        """
        Validates a model
        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation Data
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        float
            Calculated metric if a metric is provided, else None
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        self.model.eval()
        ic_loss = AverageMeter()
        ner_loss = AverageMeter()
        summary_loss = AverageMeter()
        y_preds = []
        y_true = []
        batch_size = val_loader.batch_size

        t = time.time()
        for step, data in enumerate(val_loader):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rVal Step {step}/{len(val_loader)} | ' +
                        f'ic_loss: {ic_loss.avg:.5f} | ' +
                        f'ner_loss: {ner_loss.avg:.5f} | ' +
                        f'summary_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(val_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x, y, w = self.unpack(data)

                if metric:
                    y_true += y.cpu().numpy().tolist()

                # just forward pass
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)

                loss = self.loss_function(output, y)

                ic_loss.update(loss['IC'].detach().item(), batch_size)
                ner_loss.update(loss['NER'].detach().item(), batch_size)
                summary_loss.update(loss['summary'].detach().item(), batch_size)

                if metric:
                    y_preds += output.cpu().numpy().tolist()

        # Callback metrics
        metric_log = ' '*30
        if metric:
            calculated_metrics = []
            y_pred = np.argmax(y_preds, axis=1)
            for f, args in metric:
                value = f(y_true, y_pred, **args)
                calculated_metrics.append((value, f.__name__))
                metric_log = f'- {f.__name__} {value:.5f} '
        else:
            calculated_metrics = None

        self.log(f"\r[VALIDATION] {(time.time() - t):.2f}s - "\
                 f"val ic loss: {ic_loss.avg:.5f} " + \
                 f"val ner loss: {ner_loss.avg:.5f} " + \
                 f"val summary loss: {summary_loss.avg:.5f} " + \
                 metric_log
                 )
        return {'IC':ic_loss, 'NER':ner_loss, 'summary':summary_loss}, calculated_metrics
