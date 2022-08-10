# Requirements
import torch
import spacy
import numpy as np
from typing import Dict
from transformers import AutoTokenizer

# Dataset
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
        # Extra utilities
        self.label = list(nlp.get_pipe('entity_ruler').labels)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        # Fetch text
        text = self.corpus[idx]
        # Get tokens, attention mask and NER labels
        items, ner_labels = self._collate_spaCy_HuggingFace(text)
        # Get IC labels
        ic_labels = torch.from_numpy(np.array(self.intent2idx[self.intents[idx]]))
        # Concatenate IC and NER labels to be processed in HuggingFace trainer
        # To that end, we need to reshape NER labels from (bs, max_length, labels['NER'])
        # to (bs, max_length*labels['NER'])
        return {
            'x':items,
            'y':{'IC':ic_labels, 'NER':ner_labels},
            }

    def _collate_spaCy_HuggingFace(self, text):
        # Build dictionaries
        doc = self.nlp(text)
        entlist = [(elem.label_, elem.start_char, elem.end_char) for elem in doc.ents if elem.label_ in self.label]
        # Tokenize text
        tokens = self.tokenizer.encode_plus(text,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_offsets_mapping=True,
                                            return_tensors='pt',
                                            )
    
        # Create array to store each class labels
        ## First axis indicates the label
        ## Second axis each text
        ## Third axis the token position
        targets = np.zeros((self.max_length), dtype='int32') #Everything is unlabelled by Default
    
        # FIND TARGETS IN TEXT AND SAVE IN TARGET ARRAYS
        offsets = np.squeeze(tokens['offset_mapping'].numpy())
        offset_index = 0
        for index, (label, start, end) in enumerate(entlist):
            a = int(start)
            b = int(end)
            if offset_index>len(offsets)-1:
                break
            c = offsets[offset_index][0] # Token start
            d = offsets[offset_index][1] # Token end
            count_token = 0 # token counter
            beginning = True
            while b>c: # While tokens lie in the discourse of a specific entity
                if (c>=a)&(b>=d): # If token is inside discourse
                    if beginning:
                        targets[offset_index] = self.ner2idx['B-'+label]
                        beginning = False
                    else:
                        targets[offset_index] = self.ner2idx['I-'+label]
                count_token += 1
                offset_index += 1 # Move to the next token
                if offset_index>len(offsets)-1: # If next token is out of this entity range, jump to the next row of the df
                    break
                c = offsets[offset_index][0]
                d = offsets[offset_index][1]
        # UPDATE: REMOVE -100 TOKEN
        ## 'PAD' label to make loss function ignore padding, which is basically where attn_tokens is zero
        #targets[np.where(np.array(np.squeeze(tokens['attention_mask'].numpy()))==0)[0]] = ner2idx['PAD']
        # Save in dictionary
        ner_target =  torch.LongTensor(targets)
        # End of method
        return {'input_ids': torch.squeeze(tokens['input_ids']), 'attention_mask':torch.squeeze(tokens['attention_mask'])}, ner_target