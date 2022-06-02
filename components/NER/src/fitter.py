# Requirements
from transformers import Trainer
import torch
from src.loss import FocalLoss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ner_labels = inputs.get('labels')
        # forward pass
        outputs = model(inputs.get('input_ids'), inputs.get('attention_mask'))
        # compute custom loss
        loss_fn = FocalLoss(gamma = 2., n_classes = model.num_labels)
        ner_loss = loss_fn(outputs, ner_labels)
        return (ner_loss, outputs) if return_outputs else ner_loss