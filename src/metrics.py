# Requirements
from utils import convert_tags, computeF1Score
import numpy as np
import torch
import torchmetrics


# NER
def F1_NER(ner_model_output, ner_ground_truth, idxs2tag, original_idxs2tag):
    # Model output conversion
    output0, output1 = torch.argmax(ner_model_output['H_NER']['label_0'], dim=-1).cpu().numpy(), torch.argmax(ner_model_output['H_NER']['label_1'], dim=-1).cpu().numpy()
    input = np.core.defchararray.add(np.core.defchararray.add(np.vectorize(idxs2tag['label_0'].get)(output0), np.full(output1.shape,'.')),np.vectorize(idxs2tag['label_1'].get)(output1))
    input = np.vectorize(convert_tags)(input, original_idxs2tag)
    # Ground truth conversion
    GT0, GT1 = ner_ground_truth['H_NER']['label_0'].cpu().numpy(), ner_ground_truth['H_NER']['label_1'].cpu().numpy()
    target = np.core.defchararray.add(np.core.defchararray.add(np.vectorize(idxs2tag['label_0'].get)(GT0), np.full(output1.shape,'.')),np.vectorize(idxs2tag['label_1'].get)(GT1))
    target = np.vectorize(convert_tags)(target, original_idxs2tag)
    # Metrics
    f1, precision, recall = computeF1Score(input, target)
    return f1, precision, recall
