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