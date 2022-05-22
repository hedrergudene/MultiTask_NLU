# Requirements
from .utils import *
import numpy as np
import torch
import torchmetrics

# Accuracy for text classification
class AccIC(torch.nn.Module):
    def __init__(self,
                 num_classes:int=18,
                 multi_class:bool=True,
                ):
        super(AccIC, self).__init__()
        self.acc_ic = torchmetrics.Accuracy(threshold=0.5,
                                            num_classes=num_classes,
                                            multi_class=multi_class,
                                            average="macro",
                                            mdmc_average="samplewise",
                                            )
    
    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
               ):
        return self.acc_ic(input['IC'].detach().cpu(), target['IC'].detach().cpu().type('torch.LongTensor')).item()

# F1 score for text classification
class F1IC(torch.nn.Module):
    def __init__(self,
                 num_classes:int=18,
                 multi_class:bool=True,
                ):
        super(F1IC, self).__init__()
        self.f1_ic = torchmetrics.F1Score(threshold=0.5,
                                          num_classes=num_classes,
                                          multi_class=multi_class,
                                          average="macro",
                                          mdmc_average="samplewise",
                                          )
    
    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
               ):
        return self.f1_ic(input['IC'].detach().cpu(), target['IC'].detach().cpu().type('torch.LongTensor')).item()

# Accuracy for token classification
class AccNER(torch.nn.Module):
    def __init__(self,
                 num_classes:int=167,
                 multi_class:bool=False,
                ):
        super(AccNER, self).__init__()
        self.acc_ner = torchmetrics.Accuracy(threshold=0.5,
                                             #num_classes=num_classes,
                                             ignore_index=-100,
                                             multi_class=multi_class, 
                                             #average="macro",
                                             mdmc_average="samplewise",
                                             )
    
    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
               ):
        return self.acc_ner(torch.argmax(input['NER'], dim=-1).detach().cpu(), target['NER'].detach().cpu().type('torch.LongTensor')).item()
    
# F1 score for token classification
class F1NER(torch.nn.Module):
    def __init__(self,
                 num_classes:int=167,
                 multi_class:bool=False,
                ):
        super(F1NER, self).__init__()
        self.f1_ner = torchmetrics.F1Score(threshold=0.5,
                                           #num_classes=num_classes,
                                           ignore_index=-100,
                                           multi_class=multi_class, 
                                           #average="macro",
                                           mdmc_average="samplewise",
                                           )
    
    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
               ):
        return self.f1_ner(torch.argmax(input['NER'], dim=-1).detach().cpu(), target['NER'].detach().cpu().type('torch.LongTensor')).item()
