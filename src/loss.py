# Requirements
from typing import List
import torch

class MT_IC_HNER_Loss(torch.nn.Module):
    def __init__(self,
                 entity_tags:List[str],
                 reduction:str='sum',
                 label_smoothing:float=.1,
                 ):
        super(MT_IC_HNER_Loss, self).__init__()
        # Checks
        assert reduction in ["sum","mean"], f"Reduction parameter has to be either 'sum' or 'mean'."
        # Parameters
        self.tags = entity_tags
        self.loss_fn_ent = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing = label_smoothing)
        self.reduction = reduction
        # Parameters for loss weights
        self.weight_ic_ner = torch.nn.Parameter(torch.zeros((1)))
        self.weights_ner = torch.nn.Parameter(torch.zeros((len(self.tags))))
    
    def forward(self, logits, target):
        # IC multilabel branch
        loss_IC = -torch.sum(target["IC"]*torch.nn.functional.logsigmoid(logits["IC"]), dim=-1)
        if self.reduction=='sum':
            loss_IC = torch.sum(loss_IC, dim=-1)
        else:
            loss_IC = torch.mean(loss_IC, dim=-1)
        # H_NER branch
        preds = {column:torch.transpose(torch.nn.functional.log_softmax(logits['H_NER'][column], dim=1),dim0=2,dim1=1) for column in self.tags}
        loss_H_NER = torch.stack([self.loss_fn_ent(preds[column], target['H_NER'][column]) for column in self.tags], dim=0)
        w = torch.nn.functional.sigmoid(self.weights_ner)/torch.sum(torch.nn.functional.sigmoid(self.weights_ner))
        loss_H_NER = torch.dot(w, loss_H_NER)
        return torch.nn.functional.sigmoid(self.weight_ic_ner)*loss_IC + (1-torch.nn.functional.sigmoid(self.weight_ic_ner))*loss_H_NER
