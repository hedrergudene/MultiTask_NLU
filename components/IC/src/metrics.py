# Requirements
from .utils import *
import numpy as np
from tqdm import tqdm
import logging as log
from sklearn.metrics import accuracy_score, f1_score
import torch


def evaluate_metrics(trainer, val_dtl, language_arr):
    # Setup
    idx2ner = {v:k for k,v in trainer.eval_dataset.ner2idx.items()}
    device = 'cuda' if not trainer.args.no_cuda else 'cpu'
    IC_LABELS, IC_OUTPUT = [], []
    # Create loop with custom metrics
    log.info("Stack predictions:")
    for batch in tqdm(iter(val_dtl)):
        # Get labels
        ic_labels = torch.squeeze(batch.get('labels')).detach().numpy()
        batch = {k:v.to(device) for k,v in batch.items()}
        # Get output
        with torch.no_grad():
            output = trainer.model(**batch)
        ic_output = torch.argmax(output, dim=-1).detach().cpu().numpy()
        # Append results
        IC_LABELS.append(ic_labels)
        IC_OUTPUT.append(ic_output)
    # Build final objects
    IC_LABELS = np.concatenate(IC_LABELS)
    IC_OUTPUT = np.concatenate(IC_OUTPUT)
    # Compute global metrics
    log.info("Compute global metrics:")
    accIC = accuracy_score(IC_LABELS, IC_OUTPUT)
    f1IC = f1_score(IC_LABELS, IC_OUTPUT, average='macro')
    global_metrics={'accuracy_IC':accIC,'f1_IC':f1IC}
    # Compute language-wise metrics
    log.info("Compute language-wise metrics:")
    lang_metrics={}
    for lang in tqdm(np.unique(language_arr)):
        accIC = accuracy_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang])
        f1IC = f1_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang], average='macro')
        lang_metrics[lang]={'accuracy_IC':accIC,'f1_IC':f1IC}
    return global_metrics, lang_metrics
