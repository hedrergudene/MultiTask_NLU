# Requirements
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
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(LinearBlock, self).__init__()
        # Parameters
        self.block = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                         torch.nn.Linear(input_dim, output_dim, device=device),
                                         torch.nn.LayerNorm(output_dim, device=device),
                                         torch.nn.GELU(),
                                        )
    
    def forward(self, x):
        return self.block(x)

## Format
class DataFormat(torch.nn.Module):
    def __init__(self,
                 hidden_size:int,
                 dim:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(DataFormat, self).__init__()
        # Parameters
        self.Lblock_ic = LinearBlock(hidden_size, dim, dropout, device)
        self.Lblock_ner = LinearBlock(hidden_size, dim, dropout, device)
    
    def forward(self, ic_tokens, ner_tokens):
        ic_tokens = self.Lblock_ic(ic_tokens)
        ner_tokens = self.Lblock_ner(ner_tokens)
        return ic_tokens, ner_tokens


## NER2IC
class NER2IC(torch.nn.Module):
    def __init__(self,
                 dim:int,
                 num_labels_ic:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(NER2IC, self).__init__()
        # Parameters
        self.Lblock_prior = LinearBlock(dim, 1, dropout, device)
        self.Lblock_post = LinearBlock(dim, num_labels_ic, dropout, device)
    
    def forward(self, ic_tokens, ner_tokens):
        ner_tokens = torch.mean(ner_tokens, dim=1, keepdim=True)
        ic_tokens = torch.bmm(torch.unsqueeze(ic_tokens, dim=-1), ner_tokens)
        ic_tokens = torch.squeeze(self.Lblock_prior(ic_tokens))
        ic_tokens = self.Lblock_post(ic_tokens)
        return ic_tokens

## IC2NER
class IC2NER(torch.nn.Module):
    def __init__(self,
                 dim:int,
                 num_labels_ner:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(IC2NER, self).__init__()
        # Parameters
        self.Lblock = LinearBlock(1, num_labels_ner, dropout, device)
    
    def forward(self, ic_tokens, ner_tokens):
        ic_tokens = torch.unsqueeze(ic_tokens, dim=-1)
        ic_tokens = self.Lblock(ic_tokens)
        ner_tokens = torch.bmm(ner_tokens, ic_tokens)
        return ner_tokens


# Model
class IC_NER_Model(torch.nn.Module):
    def __init__(self,
                 model_name:str,
                 num_labels:Dict,
                 dim:int=128,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(IC_NER_Model, self).__init__()
        # Parameters
        self.num_labels = num_labels
        if dim>=max(num_labels['IC'],num_labels['NER']):
            self.dim = dim
        else:
            self.dim = max(num_labels['IC'],num_labels['NER'])
            log.info(f"Dimension assigned to projection was not bigger than all number of labels. Modified to {self.dim}.")
        # Take pretrained model from custom configuration
        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": False,
            }
        )
        self.pos_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.transformer = AutoModel.from_config(config).to(device)
        # Layers
        self.LFormat = DataFormat(self.hidden_size, self.dim, dropout, device)
        self.ic_layer = NER2IC(self.dim, self.num_labels['IC'], dropout, device)
        self.ner_layer = IC2NER(self.dim, self.num_labels['NER'], dropout, device)
    
    def forward(self, tokens, attn_mask):
        # Input
        ic_tokens, ner_tokens = self._disentangle_transformer(tokens, attn_mask)
        ic_tokens, ner_tokens = self.LFormat(ic_tokens, ner_tokens)
        # Info sharing
        ner_output = self.ner_layer(ic_tokens, ner_tokens)
        ic_output = self.ic_layer(ic_tokens, ner_tokens)
        # Output
        return {'IC':ic_output,
                'NER':ner_output,
                }
    
    def _disentangle_transformer(self,
                                 tokens:torch.Tensor,
                                 attn_mask:torch.Tensor,
                                 ):
        # Get HuggingFace output
        output = self.transformer(tokens, attn_mask)
        # Check that output has the desired attribute
        if not hasattr(output, 'last_hidden_state'):
            raise AttributeError(f"Transformers output does not have the attribute 'last_hidden_state'. Please check imported backbone.")
        # Return ic_tokens and ner_tokens
        return torch.mean(output.last_hidden_state, dim=1), output.last_hidden_state
