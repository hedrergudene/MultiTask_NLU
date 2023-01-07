# Requirements
from .utils import *
import numpy as np
from tqdm import tqdm
import logging as log
import time
from sklearn.metrics import accuracy_score, f1_score
import torch


def evaluate_metrics_ONNX(ort_sess, dtl, language_arr):
    # Setup
    idx2ner = {v:k for k,v in dtl.dataset.tag2idx.items()}
    IC_LABELS, IC_OUTPUT, NER_LABELS, NER_OUTPUT = [], [], [], []
    # Create loop with custom metrics
    start_ts = time.time()
    for batch in tqdm(iter(dtl)):
        # Get labels
        ic_labels = torch.squeeze(batch['y']['IC']).detach().cpu().numpy()
        ner_labels = batch['y']['NER'].detach().cpu().numpy()
        # Get output
        outputs = ort_sess.run(None, {k:v.numpy() for k,v in batch.get('x').items()})[0]
        ic_output = torch.argmax(outputs[0], dim=-1).detach().cpu().numpy()
        ner_output = torch.argmax(outputs[1], dim=-1).detach().cpu().numpy()
        # Decode NER arrays
        ner_labels = np.vectorize(idx2ner.get)(ner_labels)
        ner_output = np.vectorize(idx2ner.get)(ner_output)
        # Append results
        IC_LABELS.append(ic_labels)
        IC_OUTPUT.append(ic_output)
        NER_LABELS.append(ner_labels)
        NER_OUTPUT.append(ner_output)
    # Build final objects
    IC_LABELS = np.concatenate(IC_LABELS)
    IC_OUTPUT = np.concatenate(IC_OUTPUT)
    NER_LABELS = np.concatenate(NER_LABELS)
    NER_OUTPUT = np.concatenate(NER_OUTPUT)
    # Compute global metrics
    log.info("Compute global metrics:")
    accIC = accuracy_score(IC_LABELS, IC_OUTPUT)
    f1_macro_IC = f1_score(IC_LABELS, IC_OUTPUT, average='macro')
    f1_weighted_IC = f1_score(IC_LABELS, IC_OUTPUT, average='weighted')
    f1NER, precision, recall = computeF1Score(NER_OUTPUT, NER_LABELS)
    global_metrics={
        'accuracy_IC':accIC,
        'f1_macro_IC':f1_macro_IC,
        'f1_weighted_IC':f1_weighted_IC,
        'f1_macro_NER':f1NER,
        'precision_NER':precision,
        'recall_NER':recall,
        }
    # Compute language-wise metrics
    log.info("Compute language-wise metrics:")
    lang_metrics={}
    for lang in tqdm(np.unique(language_arr)):
        accIC = accuracy_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang])
        f1_macro_IC = f1_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang], average='macro')
        f1_weighted_IC = f1_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang], average='weighted')
        f1NER, precision, recall = computeF1Score(NER_OUTPUT[language_arr==lang], NER_LABELS[language_arr==lang])
        lang_metrics[lang]={
            'accIC':accIC,
            'f1_macro_IC':f1_macro_IC,
            'f1_weighted_IC':f1_weighted_IC,
            'f1_NER':f1NER,
            'precision_NER':precision,
            'recall_NER':recall,
            }
    return global_metrics, lang_metrics


def evaluate_metrics(model, train_dct, val_dtl, language_arr):
    # Setup
    idx2ner = {v:k for k,v in val_dtl.dataset.ner2idx.items()}
    device = train_dct['device']
    IC_LABELS, IC_OUTPUT, NER_LABELS, NER_OUTPUT = [], [], [], []
    # Create loop with custom metrics
    log.info("Stack predictions:")
    for batch in tqdm(iter(val_dtl)):
        # Get labels
        ic_labels = torch.squeeze(batch['y']['IC']).detach().cpu().numpy()
        ner_labels = batch['y']['NER'].detach().cpu().numpy()
        batch = {k:v.to(device) for k,v in batch['x'].items()}
        # Get output
        with torch.no_grad():
            output = model(**batch)
        ic_output = torch.argmax(output[0], dim=-1).detach().cpu().numpy()
        ner_output = torch.argmax(output[1], dim=-1).detach().cpu().numpy()
        # Decode NER arrays
        ner_labels = np.vectorize(idx2ner.get)(ner_labels)
        ner_output = np.vectorize(idx2ner.get)(ner_output)
        # Append results
        IC_LABELS.append(ic_labels)
        IC_OUTPUT.append(ic_output)
        NER_LABELS.append(ner_labels)
        NER_OUTPUT.append(ner_output)
    # Build final objects
    IC_LABELS = np.concatenate(IC_LABELS)
    IC_OUTPUT = np.concatenate(IC_OUTPUT)
    NER_LABELS = np.concatenate(NER_LABELS)
    NER_OUTPUT = np.concatenate(NER_OUTPUT)
    # Compute global metrics
    log.info("Compute global metrics:")
    accIC = accuracy_score(IC_LABELS, IC_OUTPUT)
    f1_macro_IC = f1_score(IC_LABELS, IC_OUTPUT, average='macro')
    f1_weighted_IC = f1_score(IC_LABELS, IC_OUTPUT, average='weighted')
    f1NER, precision, recall = computeF1Score(NER_OUTPUT, NER_LABELS)
    global_metrics={
        'accuracy_IC':accIC,
        'f1_macro_IC':f1_macro_IC,
        'f1_weighted_IC':f1_weighted_IC,
        'f1_macro_NER':f1NER,
        'precision_NER':precision,
        'recall_NER':recall,
        }
    # Compute language-wise metrics
    log.info("Compute language-wise metrics:")
    lang_metrics={}
    for lang in tqdm(np.unique(language_arr)):
        accIC = accuracy_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang])
        f1_macro_IC = f1_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang], average='macro')
        f1_weighted_IC = f1_score(IC_LABELS[language_arr==lang], IC_OUTPUT[language_arr==lang], average='weighted')
        f1NER, precision, recall = computeF1Score(NER_OUTPUT[language_arr==lang], NER_LABELS[language_arr==lang])
        lang_metrics[lang]={
            'accIC':accIC,
            'f1_macro_IC':f1_macro_IC,
            'f1_weighted_IC':f1_weighted_IC,
            'f1_NER':f1NER,
            'precision_NER':precision,
            'recall_NER':recall,
            }
    return global_metrics, lang_metrics