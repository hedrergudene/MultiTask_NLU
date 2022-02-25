# Requirements
import numpy as np
import torch
from typing import List, Dict, Iterable
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
             'H_NER':{column:data[1]['H_NER'][column].to(self.device) for column in tags},
             }

        if 'w' in data:
            w = data['w'].to(self.device).float()
        else:
            w = None

        return x, y, w


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
