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
    IC_LABELS, IC_OUTPUT, NER_LABELS, NER_OUTPUT = [], [], [], []
    # Create loop with custom metrics
    log.info("Stack predictions:")
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
        # Append results
        IC_LABELS.append(ic_labels)
        IC_OUTPUT.append(ic_output)
        NER_LABELS.append(ner_labels)
        NER_OUTPUT.append(ner_output)
    # Compute metrics
    log.info("Compute metrics:")
    accIC = accuracy_score(np.concatenate(IC_LABELS), np.concatenate(IC_OUTPUT))
    f1IC = f1_score(np.concatenate(IC_LABELS), np.concatenate(IC_OUTPUT), average='macro')
    f1NER, precision, recall = computeF1Score(np.concatenate(NER_OUTPUT), np.concatenate(NER_LABELS))
    return {'accuracy_IC':accIC,
            'f1_IC':f1IC,
            'f1_NER':f1NER,
            'precision_NER':precision,
            'recall_NER':recall,
            }
