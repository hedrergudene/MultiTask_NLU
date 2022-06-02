# Requirements
from .utils import *
import numpy as np
from tqdm import tqdm
import logging as log
import torch


def evaluate_metrics(trainer, val_dtl, language_arr):
    # Setup
    idx2ner = {v:k for k,v in trainer.eval_dataset.ner2idx.items()}
    device = 'cuda' if not trainer.args.no_cuda else 'cpu'
    NER_LABELS, NER_OUTPUT = [], []
    # Create loop with custom metrics
    log.info("Stack predictions:")
    for batch in tqdm(iter(val_dtl)):
        # Get labels
        ner_labels = batch.get('labels').detach().numpy()
        batch = {k:v.to(device) for k,v in batch.items()}
        # Get output
        with torch.no_grad():
            ner_output = trainer.model(**batch)
        ner_output = torch.argmax(ner_output, dim=-1).detach().cpu().numpy()
        # Decode NER arrays
        ner_labels = np.vectorize(idx2ner.get)(ner_labels)
        ner_output = np.vectorize(idx2ner.get)(ner_output)
        # Append results
        NER_LABELS.append(ner_labels)
        NER_OUTPUT.append(ner_output)
    # Build final objects
    NER_LABELS = np.concatenate(NER_LABELS)
    NER_OUTPUT = np.concatenate(NER_OUTPUT)
    # Compute global metrics
    log.info("Compute global metrics:")
    f1NER, precision, recall = computeF1Score(NER_OUTPUT, NER_LABELS)
    global_metrics={'f1_NER':f1NER,'precision_NER':precision,'recall_NER':recall}
    # Compute language-wise metrics
    log.info("Compute language-wise metrics:")
    lang_metrics={}
    for lang in tqdm(np.unique(language_arr)):
        f1NER, precision, recall = computeF1Score(NER_OUTPUT[language_arr==lang], NER_LABELS[language_arr==lang])
        lang_metrics[lang]={'f1_NER':f1NER,'precision_NER':precision,'recall_NER':recall}
    return global_metrics, lang_metrics
