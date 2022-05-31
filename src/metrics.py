# Requirements
from .utils import *
import numpy as np
from tqdm import tqdm
import logging as log
from sklearn.metrics import accuracy_score, f1_score
import torch


def evaluate_metrics(trainer):
    # Setup
    idx2ner = {v:k for k,v in trainer.eval_dataset.ner2idx.items()}
    device = 'cuda' if not trainer.args.no_cuda else 'cpu'
    val_dtl=torch.utils.data.DataLoader(trainer.eval_dataset,
                                        batch_size=trainer.args.per_device_eval_batch_size,
                                        num_workers=trainer.args.dataloader_num_workers,
                                        )
    accIC, f1IC, f1NER, prNER, recNER = [], [], [], [], []
    # Create loop with custom metrics
    log.info("Compute metrics:")
    for batch in tqdm(iter(val_dtl)):
        # Get labels
        ic_labels = torch.squeeze(batch.get('labels')[:,:1]).detach().numpy()
        ner_labels = batch.get('labels')[:,1:].detach().numpy()
        batch = {k:v.to(device) for k,v in batch.items()}
        # Get output
        with torch.no_grad():
            output = trainer.model(**batch)
        ic_output = torch.argmax(output[:,:trainer.model.num_labels['IC']], dim=-1).detach().cpu().numpy()
        ner_output = output[:,trainer.model.num_labels['IC']:].reshape((-1, trainer.model.max_length, trainer.model.num_labels['NER']))
        ner_output = torch.argmax(ner_output, dim=-1).detach().cpu().numpy()
        # Decode NER arrays
        ner_labels = np.vectorize(idx2ner.get)(ner_labels)
        ner_output = np.vectorize(idx2ner.get)(ner_output)
        # Get metrics
        accIC.append(accuracy_score(ic_labels, ic_output))
        f1IC.append(f1_score(ic_labels, ic_output, average='macro'))
        f1s, precision, recall = computeF1Score(ner_output, ner_labels)
        f1NER.append(f1s)
        prNER.append(precision)
        recNER.append(recall)
    return {'accuracy_IC':np.mean(accIC),
            'f1_IC':np.mean(f1IC),
            'f1_NER':np.mean(f1NER),
            'precision_NER':np.mean(prNER),
            'recall_NER':np.mean(recNER),
            }
