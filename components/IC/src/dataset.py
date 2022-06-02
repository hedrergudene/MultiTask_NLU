# Requirements
import torch
import numpy as np
from typing import Dict
from transformers import AutoTokenizer

# Dataset
class IC_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 corpus,
                 intents,
                 model_name:str,
                 max_length:int,
                 intent2idx:Dict,
                 ):
        # Parameters
        self.corpus = corpus
        self.intents = intents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.intent2idx = intent2idx

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        # Fetch text
        text = self.corpus[idx]
        # Get tokens, attention mask and NER labels
        items = self._collate_HuggingFace(text)
        # Get IC labels
        ic_labels = torch.from_numpy(np.array(self.intent2idx[self.intents[idx]]))
        # Concatenate IC and NER labels to be processed in HuggingFace trainer
        # To that end, we need to reshape NER labels from (bs, max_length, labels['NER'])
        # to (bs, max_length*labels['NER'])
        items['labels'] = ic_labels
        return items

    def _collate_HuggingFace(self, text):
        # Tokenize text
        tokens = self.tokenizer.encode_plus(text,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_offsets_mapping=True,
                                            return_tensors='pt',
                                            )
        # End of method
        return {'input_ids': torch.squeeze(tokens['input_ids']), 'attention_mask':torch.squeeze(tokens['attention_mask'])}
