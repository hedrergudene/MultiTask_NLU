# Requirements
from .utils import *
import numpy as np
import torch
import torchmetrics

# IC
def Metrics_IC(model_output, ground_truth, ic_metrics_kwargs):
    acc_ml = torchmetrics.Accuracy(**ic_metrics_kwargs)
    f1_ml = torchmetrics.F1Score(**ic_metrics_kwargs)
    return [(acc_ml(model_output['IC'].detach().cpu(), ground_truth['IC'].detach().cpu()).item(), 'acc_IC'),
            (f1_ml(model_output['IC'].detach().cpu(), ground_truth['IC'].detach().cpu()).item(), 'f1_IC'),
           ]

# H_NER metrics
def Metrics_NER(model_output, ground_truth, idxs2tag, original_idxs2tag):
    # Model output conversion
    output0, output1 = torch.argmax(model_output['H_NER']['label_0'].detach(), dim=-1).cpu().numpy(), torch.argmax(model_output['H_NER']['label_1'].detach(), dim=-1).cpu().numpy()
    input = np.core.defchararray.add(np.core.defchararray.add(np.vectorize(idxs2tag['label_0'].get)(output0), np.full(output1.shape,'.')),np.vectorize(idxs2tag['label_1'].get)(output1))
    input = np.vectorize(convert_tags)(input, original_idxs2tag)
    # Ground truth conversion
    GT0, GT1 = ground_truth['H_NER']['label_0'].detach().cpu().numpy(), ground_truth['H_NER']['label_1'].detach().cpu().numpy()
    target = np.core.defchararray.add(np.core.defchararray.add(np.vectorize(idxs2tag['label_0'].get)(GT0), np.full(output1.shape,'.')),np.vectorize(idxs2tag['label_1'].get)(GT1))
    target = np.vectorize(convert_tags)(target, original_idxs2tag)
    # Metrics
    f1, precision, recall = computeF1Score(input, target)
    return [(f1,'f1_NER'), (precision,'precision_NER'), (recall,'recall_NER')]

# Combine methods
def metrics(input, target, ic_metrics_kwargs, idxs2tag, original_idxs2tag):
    return Metrics_IC(input, target, ic_metrics_kwargs) + Metrics_NER(input, target, idxs2tag, original_idxs2tag)
