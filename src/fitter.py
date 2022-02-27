# Requirements
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Iterable, Callable, Tuple
from datetime import datetime
import time
from benatools.torch.fitter import AverageMeter, TorchFitterBase

# Fitter method
class MT_IC_HNER_Fitter(TorchFitterBase):

    def __init__(self,
                 ic_metrics_kwargs,
                 idxs2tag,
                 original_idxs2tag,
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

        self.ic_metrics_kwargs = ic_metrics_kwargs
        self.idxs2tag = idxs2tag
        self.tags = list(self.idxs2tag.keys())
        self.original_idxs2tag = original_idxs2tag
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


    # Mechanism to unwrap data and place it in the right device
    def unpack(self, data):
        x = {k: v.to(self.device) for k, v in data[0].items()}

        y = {'IC':data[1]['IC'].to(self.device),
             'H_NER':{column:data[1]['H_NER'][column].to(self.device) for column in self.tags},
             }

        if 'w' in data:
            w = data['w'].to(self.device).float()
        else:
            w = None

        return x, y, w

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
                try:
                    self.scheduler.step(metrics=calculated_metric)
                except:
                    self.scheduler.step()

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')



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

        # Callback metrics
        metric_log = ' '*30
        if metric:
            calculated_metrics = metric(output, y, self.ic_metrics_kwargs, self.idxs2tag, self.original_idxs2tag)
            metric_log = ' '.join([f'- {name} {value:.5f}' for value, name in calculated_metrics])
        else:
            calculated_metrics = None

        self.log(f'\r[VALIDATION] {(time.time() - t):.2f}s - val. loss: {summary_loss.avg:.5f} ' + metric_log)
        return summary_loss, calculated_metrics
