import torch
from typing import Dict
from transformers import AutoTokenizer
from .utils import collate_spaCy_HuggingFace

class MT_IC_HNER_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 corpus,
                 intents,
                 model_name:str,
                 max_length:int,
                 tag2nlp:Dict,
                 tag2idx:Dict,
                 ):
        # Parameters
        self.corpus = corpus
        self.intents = intents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.tag2nlp = tag2nlp
        self.tag2idx = tag2idx
        self.tags = list(self.tag2nlp.keys())
        # Extra utilities
        self.tag2label = {column:list(set([elem[2:] for elem in self.tag2idx[column].keys() if elem[:2] in ['B-','I-']]+["PAD"])) for column in self.tags}

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        intent = torch.LongTensor(self.intents[idx,:])
        input, tag2target = collate_spaCy_HuggingFace(text, self.tag2nlp, self.tokenizer, self.max_length, self.tag2idx, self.tag2label)
        return input, {"IC":intent, "H_NER":tag2target}