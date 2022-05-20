import torch
from typing import Dict
from transformers import AutoTokenizer
from .utils import collate_spaCy_HuggingFace

class IC_NER_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 corpus,
                 intents,
                 model_name:str,
                 max_length:int,
                 nlp,
                 intent2idx:Dict,
                 ner2idx:Dict,
                 ):
        # Parameters
        self.corpus = corpus
        self.intents = intents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.nlp = nlp
        self.intent2idx = intent2idx
        self.ner2idx = ner2idx
        self.device = device
        # Extra utilities
        self.label = list(nlp.get_pipe('entity_ruler').labels)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        intent = self._multilabel_intent(self.intents[idx].split("+"))
        input, target = collate_spaCy_HuggingFace(text, self.nlp, self.tokenizer, self.max_length, self.tag2idx, self.label)
        return input, {"IC":intent, "NER":target}

    def _multilabel_intent(self, labels):
        target = torch.zeros((1,len(self.intent2idx)))
        for label in labels:
            target[0,self.intent2idx[label]] = 1
        return target.type('torch.LongTensor')
