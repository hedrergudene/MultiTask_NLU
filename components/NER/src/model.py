# Requirements
import logging as log
from typing import Dict, List
import numpy as np
import torch
from transformers import AutoConfig, AutoModel

# Utils

## Linear Block
class LinearBlock(torch.nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 activation:bool=True,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(LinearBlock, self).__init__()
        # Parameters
        if activation:
            self.block = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                             torch.nn.Linear(input_dim, output_dim, device=device),
                                             torch.nn.LayerNorm(output_dim, device=device),
                                             torch.nn.GELU(),
                                            )
        else:
            self.block = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                             torch.nn.Linear(input_dim, output_dim, device=device),
                                            )

    def forward(self, x):
        return self.block(x)




# Model
class NER_Model(torch.nn.Module):
    def __init__(self,
                 model_name:str,
                 max_length:int,
                 num_labels:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(NER_Model, self).__init__()
        # Parameters
        self.max_length = max_length
        self.num_labels = num_labels
        # Take pretrained model from custom configuration
        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": False,
            }
        )
        self.hidden_size = config.hidden_size
        self.transformer = AutoModel.from_config(config).to(device)
        # Layers
        self.LogitsLayer = LinearBlock(input_dim=self.hidden_size,
                                       output_dim=self.num_labels,
                                       activation=True,
                                       dropout=dropout,
                                       device=device,
                                       )
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Input
        ner_tokens = self._disentangle_transformer(input_ids, attention_mask)
        # Output
        ner_tokens = self.LogitsLayer(ner_tokens)
        return ner_tokens
    
    def _disentangle_transformer(self,
                                 input_ids:torch.Tensor,
                                 attention_mask:torch.Tensor,
                                 ):
        # Get HuggingFace output
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Check that output has the desired attribute
        if not hasattr(output, 'last_hidden_state'):
            raise AttributeError(f"Transformers output does not have the attribute 'last_hidden_state'. Please check imported backbone.")
        # Return ic_tokens and ner_tokens
        return output.last_hidden_state
