# Requirements
from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ic_labels = torch.squeeze(inputs.get('labels'))
        # forward pass
        outputs = model(inputs.get('input_ids'), inputs.get('attention_mask'))
        # compute custom loss
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.1)
        ic_loss = loss_fn(outputs, ic_labels)
        return (ic_loss, outputs) if return_outputs else ic_loss