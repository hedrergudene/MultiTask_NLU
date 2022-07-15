
# Requirements
from typing import List, Dict
import torch

# CE loss with weights
class CELoss(torch.nn.Module):
    """
    Implementation of CrossEntropyLoss minimisation function with class weighting and
    multilabel features.
    """
    def __init__(self,
                 temperature:float=1.,
                 from_logits:bool = True,
                 multilabel:bool=False,
                 label_smoothing:float=.05,
                 keep_sum:bool=True,
                 reduction:str = 'sum',
                 n_classes:int = None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda',
                 )->None:

        super(CELoss, self).__init__()
        # Validations
        if not torch.is_tensor(class_weights):
            raise TypeError("Class weights type is not a torch.Tensor. Got {}"
                            .format(type(class_weights)))
        if class_weights is not None:
            if len(class_weights.shape)!=2:
                raise TypeError("Class weights do not have the right shape. Got shape {}"
                                .format(len(class_weights.shape)))
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction should be one of these values: {}"
                            .format(', '.join(['mean', 'sum', 'none'])))
        # Loss config settings
        self.from_logits = from_logits
        self.multilabel = multilabel
        self.label_smoothing = label_smoothing
        self.keep_sum = keep_sum
        self.reduction = reduction
        # Loss parameters
        self.temperature = temperature
        self.class_weights = torch.ones((n_classes)).unsqueeze(dim=-1).to(device) if class_weights is None else class_weights.to(device)
        self.n_classes = n_classes

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
        ## One-hot encode labels
        if len(target.shape) < len(input.shape): target = torch.nn.functional.one_hot(target, num_classes=self.n_classes).float()
        ## Label smoothing
        if self.label_smoothing > 0: target = self.label_smoothing_fn(self.label_smoothing, target)

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


    def label_smoothing_fn(self,
                           ls:float,
                           target:torch.Tensor,
                           )->torch.Tensor:
        target = target.clone()
        if self.keep_sum:
            bias = torch.sum(target, axis = 1)*ls/(target.shape[1]-torch.sum(target, axis = 1))
            slope = ((1-ls)-bias)
        else:
            bias = torch.ones(target.shape[0])*ls/(target.shape[1]-torch.ones(target.shape[0]))
            slope = ((1-ls)-bias)
        target = torch.multiply(slope.unsqueeze(dim = 1),target) + bias.unsqueeze(dim = 1)
        return target


    def compute_loss(self,
                    input:torch.Tensor,
                    target:torch.Tensor,
                    )->torch.Tensor:
        if self.from_logits:
            input_norm = torch.nn.functional.logsigmoid(input) if self.multilabel else torch.nn.functional.log_softmax(input/self.temperature, dim=-1)
        else:
            input_norm = torch.log(input)
        # Compute the actual loss and weights classes
        loss = (- target * input_norm) @ self.class_weights
        return loss


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
            if len(class_weights.shape)!=2:
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
        self.class_weights = torch.ones((n_classes)).unsqueeze(dim=-1).to(device) if class_weights is None else class_weights.to(device)
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


# Distilation loss
class DistilLoss(torch.nn.Module):
    """
    Implementation of Knowledge Distillation minimisation function.
    """
    def __init__(self,
                 teacher_model:torch.nn.Module,
                 teacher_checkpoint:str,
                 alpha:float=.1,
                 temperature:float=1.,
                 label_smoothing:float=.0,
                 from_logits:bool=True,
                 multilabel:bool=False,
                 reduction:str='sum',
                 n_classes:int=None,
                 class_weights:torch.Tensor=None,
                 device:str='cuda:0',
                 )->None:
        """_summary_

        Args:
            teacher_model (torch.nn.Module): Pretrained model to get probability distributions.
            alpha (float, optional): Teacher-labels distribtion tradeoff parameter. Defaults to .1.
            temperature (float, optional): Smoothing factor to be applied to logits when comapred
                                           to real distribution. Defaults to 1.
            label_smoothing (float, optional): Regularisation factor to be applied to logits when
                                               comapred to real distribution. Defaults to .05.
            from_logits (bool, optional): Whether input values (and teacher input values) have
                                          been applied softmax or not. Defaults to True.
            multilabel (bool, optional): _description_. Defaults to False.
            reduction (str, optional): _description_. Defaults to 'sum'.
            n_classes (int, optional): _description_. Defaults to None.
            class_weights (torch.Tensor, optional): _description_. Defaults to None.
        """
        super(DistilLoss, self).__init__()
        # Validations
        if ((n_classes is None) and (class_weights is None)):
            raise TypeError("You must provide either n_classes or class_weights.")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction should be one of these values: {}"
                            .format(', '.join(['mean', 'sum', 'none'])))
        # Loss parameters
        self.teacher_model = teacher_model
        self.teacher_model.load_state_dict(torch.load(teacher_checkpoint))
        self.teacher_model.to(device).eval()
        self.alpha = alpha
        self.from_logits = from_logits
        class_weights = torch.ones((n_classes)).unsqueeze(dim=-1).to(device) if class_weights is None else class_weights.to(device)
        # Helper methods
        self.student_loss = CELoss(temperature = temperature,
                                   class_weights = class_weights,
                                   multilabel = multilabel,
                                   from_logits = self.from_logits,
                                   reduction = reduction,
                                   label_smoothing = label_smoothing,
                                   n_classes = n_classes,
                                   device=device,
                                   )
        self.distil_loss = torch.nn.KLDivLoss(reduction = reduction)

    def forward(self, 
                teacher_tokens:Dict,
                student_input:torch.Tensor,
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
                
        loss = self.compute_loss(teacher_tokens, student_input, target)
        return loss


    def compute_loss(self,
                     teacher_tokens:Dict,
                     input:torch.Tensor,
                     target:torch.Tensor,
                     )->torch.Tensor:
        # Get teacher inference
        with torch.no_grad():
            input_teacher = self.teacher_model(input_ids=teacher_tokens['input_ids'], attention_mask=teacher_tokens['attention_mask'])
            teacher_output_ic = input_teacher[:,:input.shape[-1]]
        # Compute both target and distribution loss
        target_loss = self.student_loss(input, target)
        ## QUESTION: Does input learn better teacher distribution from logits or from prob dist?
        if self.from_logits:
            distr_loss = self.distil_loss(torch.nn.functional.softmax(input, dim=-1),
                                          torch.nn.functional.softmax(teacher_output_ic, dim=-1),
                                          )
        else:
            distr_loss = self.distil_loss(input,
                                          teacher_output_ic,
                                          )            
        # Return final loss
        loss = self.alpha*distr_loss+(1-self.alpha)*target_loss
        return loss