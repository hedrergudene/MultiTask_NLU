# Requirements
from typing import List, Dict
import torch

# Focal Loss
class FocalLoss(torch.nn.Module):
    """
    Implementation of FocalLoss minimisation function with class weighting and
    multilabel features.
    """
    def __init__(self,
                 gamma:float=1.,
                 temperature:float=1.,
                 from_logits:bool = True,
                 multilabel:bool=False,
                 reduction:str = 'mean',
                 n_classes:int = None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda',
                 )->None:
        """
        Args:
        """
        super(FocalLoss, self).__init__()
        # Validations
        if not torch.is_tensor(class_weights) and (class_weights is not None):
            raise TypeError("Class weights type is not a torch.Tensor. Got {}"
                            .format(type(class_weights)))
        if class_weights is not None:
            if len(class_weights.shape)!=1:
                raise TypeError("Class weights do not have the right shape. Got shape {}"
                                .format(len(class_weights.shape)))
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction should be one of these values: {}"
                            .format(', '.join(['mean', 'sum', 'none'])))
        # Loss config settings
        self.from_logits=from_logits
        self.multilabel = multilabel
        self.reduction=reduction
        # Loss parameters
        self.gamma = gamma
        self.temperature = temperature
        self.class_weights = torch.ones((n_classes)).unsqueeze(dim=-1).to(device) if class_weights is None else class_weights.unsqueeze(dim=-1).to(device)
        self.n_classes = n_classes
        self.eps = 1e-6

    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
                )->torch.Tensor:
        """
        Calculates the loss.

        Args:
            input (torch.Tensor): Batch of model predictions. The last dimension must contain the proability distribution for each
                                  of the classes; i.e., input shape=(batch_size, n_classes) for IntentClassification problems, and
                                  input shape=(batch_size, max_postion_embeddings, n_classes) for NER problems.
            target (torch.Tensor): Batch containing ground truth, either in shape of binarised or one-hot encoded labels.

        Returns:
            torch.Tensor: Loss tensor. If there is any reduction, output is 0-dimensional. If there is no reduction, loss is provided
                          element-wise through the batch.
        """
        # Part I: Validations
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))

        # Part II: Labels preprocessing
        # One-hot encode labels
        if len(target.shape) < len(input.shape): target = torch.nn.functional.one_hot(target, num_classes=self.n_classes).float()

        # Part III: Compute loss
        loss = self.compute_loss(input, target)

        # Part IV:  Apply reduction method
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction=='sum':
            return torch.sum(loss)
        else: # Sum is already done in class weighting
            #loss = torch.sum(loss, dim = -1)
            return loss

    def compute_loss(self,
                     input:torch.Tensor,
                     target:torch.Tensor,
                     )->torch.Tensor:
        if self.from_logits:
            input_norm = torch.nn.functional.logsigmoid(input) if self.multilabel else torch.nn.functional.log_softmax(input/self.temperature, dim=-1)
            input_soft = torch.sigmoid(input)+self.eps if self.multilabel else torch.nn.functional.softmax(input/self.temperature, dim=-1)+self.eps
        else:
            input_norm = torch.log(input)
            input_soft = input+self.eps
        # Compute the actual focal loss and weights classes
        focal_weight = torch.pow(1. - input_soft, self.gamma)
        focal_weights = - target * focal_weight * input_norm
        if len(focal_weights.shape)<3:
            focal_weights = torch.unsqueeze(focal_weights, dim=1)
        focal_loss = torch.bmm(focal_weights, self.class_weights.repeat(input.shape[0],1,1))
        return torch.squeeze(focal_loss)

# IC_NER_Loss Loss
class IC_NER_Loss(torch.nn.Module):
    """
    Implementation of IC_NER_Loss minimisation function with class weighting and
    multilabel features.
    """
    def __init__(self,
                 loss_type:str,
                 gamma:float=1.,
                 temperature:float=1.,
                 label_smoothing:float=.1,
                 from_logits:bool = True,
                 multilabel:bool= True,
                 reduction:str = 'mean',
                 n_classes:Dict = None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda',
                 )->None:
        """
        Args:
        """
        super(IC_NER_Loss, self).__init__()
        # Validations
        assert loss_type in ['CELoss', 'FocalLoss', 'Mixed'], f"Loss type should either be 'CELoss', 'FocalLoss' or 'Mixed'."
        # Loss config settings
        self.alpha = torch.nn.Parameter(torch.zeros((1)), requires_grad=True).to(device)
        if loss_type=='CELoss':
            self.loss_fn=torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
        elif loss_type=='FocalLoss':
            self.loss_ic=FocalLoss(gamma, temperature, from_logits, multilabel, reduction, n_classes['IC'], class_weights, device)
            self.loss_ner=FocalLoss(gamma, temperature, from_logits, multilabel, reduction, n_classes['NER'], class_weights, device)
        else:
            self.loss_ic=torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
            self.loss_ner=FocalLoss(gamma, temperature, from_logits, multilabel, reduction, n_classes['NER'], class_weights, device)
        # Parameters
        self.loss_type = loss_type
                    

    def forward(self,
                input:torch.Tensor,
                target:torch.Tensor,
                )->torch.Tensor:
        """
        Calculates the loss.

        Args:
            input (torch.Tensor): Batch of model predictions. The last dimension must contain the proability distribution for each
                                  of the classes; i.e., input shape=(batch_size, n_classes) for IntentClassification problems, and
                                  input shape=(batch_size, max_postion_embeddings, n_classes) for NER problems.
            target (torch.Tensor): Batch containing ground truth, either in shape of binarised or one-hot encoded labels.

        Returns:
            torch.Tensor: Loss tensor. If there is any reduction, output is 0-dimensional. If there is no reduction, loss is provided
                          element-wise through the batch.
        """
        if self.loss_type=='CELoss':
            ic_loss = self.loss_fn(input[0], target['IC'])
            ner_loss = self.loss_fn(torch.permute(input[1], (0,2,1)), target['NER'])
        else:
            ic_loss = self.loss_ic(input[0], target['IC'])
            ner_loss = self.loss_ner(input[1], target['NER'])
        summary_loss = torch.sigmoid(self.alpha)*ic_loss + (1-torch.sigmoid(self.alpha))*ner_loss
        return {'IC':ic_loss, 'NER':ner_loss, 'summary':summary_loss}