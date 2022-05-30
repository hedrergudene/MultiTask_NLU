# Requirements
from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Data comes in ensembled format (bs, 1+max_length) to combine both
        # IC and NER labels in one tensor, respectively. Therefore, first step
        # is to disentangle it
        ic_labels = torch.squeeze(inputs.get('labels')[:,:1])
        ner_labels = inputs.get('labels')[:,1:]
        # forward pass
        outputs = model(inputs.get('input_ids'), inputs.get('attention_mask'))
        ic_logits = outputs[:,:model.num_labels['IC']]
        ner_logits = outputs[:,model.num_labels['IC']:].reshape((-1, model.max_length, model.num_labels['NER']))
        # compute custom loss
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.1)
        ic_loss = loss_fn(ic_logits, ic_labels)
        ner_loss = loss_fn(ner_logits.permute((0,2,1)), ner_labels)
        return (.5*(ic_loss+ner_loss), outputs) if return_outputs else .5*(ic_loss+ner_loss)
