
# Requirements
import torch
import spacy
import numpy as np
from typing import Dict
from transformers import XLMRobertaTokenizerFast

# Dataset
class KDDataset(torch.utils.data.Dataset):
    def __init__(self,
                 corpus,
                 intents,
                 teacher_model_name:str,
                 student_model_name:str,
                 max_length:Dict,
                 intent2idx:Dict,
                 ):
        # Parameters
        self.corpus = corpus
        self.intents = intents
        self.teacher_tokenizer = XLMRobertaTokenizerFast.from_pretrained(teacher_model_name)
        self.student_tokenizer = XLMRobertaTokenizerFast.from_pretrained(student_model_name)
        self.max_length = max_length
        self.intent2idx = intent2idx

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        # Fetch text
        text = self.corpus[idx]
        # Get tokens, attention mask and NER labels
        teacher_items = self._collate_HuggingFace(text, self.teacher_tokenizer, 'teacher')
        student_items = self._collate_HuggingFace(text, self.student_tokenizer, 'student')
        items = {**teacher_items, **student_items}
        # Get IC labels
        ic_labels = torch.from_numpy(np.array(self.intent2idx[self.intents[idx]]))
        items['labels'] = ic_labels
        return items

    def _collate_HuggingFace(self, text, tok, flag):
        # Tokenize text
        tokens = tok.encode_plus(text,
                                 max_length=self.max_length[flag],
                                 padding='max_length',
                                 truncation=True,
                                 return_offsets_mapping=True,
                                 return_tensors='pt',
                                 )
        # End of method
        if flag=='teacher':
            return {flag+'_input_ids': torch.squeeze(tokens['input_ids']), flag+'_attention_mask':torch.squeeze(tokens['attention_mask'])}
        else:
            return {'input_ids': torch.squeeze(tokens['input_ids']), 'attention_mask':torch.squeeze(tokens['attention_mask'])}