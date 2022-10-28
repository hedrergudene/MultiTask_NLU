# Requirements
import logging as log
from typing import Dict, List
import numpy as np
import torch
from transformers import AutoConfig, AutoModel

# Utils

## Linear Block
class LinearBlock(torch.nn.Module):
    """
    Auxiliary layer to comprise a standard linear block
    """
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 activation:bool=True,
                 dropout:float=.25,
                 device:str='cuda:0',
                 )->None:
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

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.block(x)


## Format
class DataFormat(torch.nn.Module):
    """
    Auxiliary layer to standardise IC and NER tokens shape, and perform 
    independent processing before information sharing.
    """
    def __init__(self,
                 hidden_size:int,
                 dim:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 )->None:
        super(DataFormat, self).__init__()
        # Parameters
        self.Lblock_ic = LinearBlock(hidden_size, dim, True, dropout, device)
        self.Lblock_ner = LinearBlock(hidden_size, dim, True, dropout, device)
    
    def forward(self, ic_tokens:torch.Tensor, ner_tokens:torch.Tensor)->torch.Tensor:
        """Standardisation module in which hidden_size is transformed into 'dim' length.

        Args:
            ic_tokens (torch.Tensor): IC tokens with shape (batch_size, hidden_size)
            ner_tokens (torch.Tensor): NER tokens with shape (bs, max_pos_emb, hidden_size)

        Returns:
            ic_tokens (torch.Tensor): Formatted IC tokens with shape (batch_size, dim)
            ner_tokens (torch.Tensor): Formatted NER tokens with shape (bs, max_pos_emb, dim)
        """
        ic_tokens = self.Lblock_ic(ic_tokens)
        ner_tokens = self.Lblock_ner(ner_tokens)
        return ic_tokens, ner_tokens


## NER2IC
class NER2IC(torch.nn.Module):
    """
    Information-sharing layer from NER branch to IC one.
    """
    def __init__(self,
                 dim:int,
                 max_length:int,
                 num_labels_ic:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(NER2IC, self).__init__()
        # Parameters
        self.Lblock_prior = LinearBlock(dim, dim//2, True, dropout, device)
        self.Lblock_post = LinearBlock(dim//2, num_labels_ic, True, dropout, device)
    
    def forward(self, ic_tokens:torch.Tensor, ner_tokens:torch.Tensor)->torch.Tensor:
        """Information sharing module. NER tokens are averaged 'horizontally' to be combined with
            IC information. As a result of that multiplication, we get an attention matrix that 
            comprises information for each token and each class.

        Args:
            ic_tokens (torch.Tensor): Formatted IC tokens with shape (batch_size, dim)
            ner_tokens (torch.Tensor): Formatted NER tokens with shape (bs, max_pos_emb, dim)

        Returns:
            torch.Tensor: Output for IC classifier
        """
        ner_tokens = torch.mean(ner_tokens, dim=1, keepdim=True)
        ic_tokens = torch.bmm(torch.unsqueeze(ic_tokens, dim=-1), ner_tokens)
        ic_tokens = torch.mean(self.Lblock_prior(ic_tokens), dim=1)
        ic_tokens = self.Lblock_post(ic_tokens)
        return ic_tokens


## IC2NER
class IC2NER(torch.nn.Module):
    """
    Information-sharing layer from IC branch to entity recognition one.
    """
    def __init__(self,
                 dim:int,
                 num_labels_ner:int,
                 dropout:float=.25,
                 device:str='cuda:0',
                 )->None:
        super(IC2NER, self).__init__()
        # Parameters
        self.dim = dim
        self.Lblock_post = LinearBlock(self.dim, num_labels_ner, True, dropout, device)
    
    def forward(self, ic_tokens:torch.Tensor, ner_tokens:torch.Tensor)->torch.Tensor:
        """Information sharing module. IC tokens are...

        Args:
            ic_tokens (torch.Tensor): Formatted IC tokens with shape (batch_size, dim)
            ner_tokens (torch.Tensor): Formatted NER tokens with shape (bs, max_pos_emb, dim)

        Returns:
            torch.Tensor: Output for NER classifier
        """
        ic_tokens = torch.unsqueeze(torch.mean(ic_tokens, dim=1, keepdim=True), dim=-1)
        ner_output = ner_tokens*ic_tokens
        ner_output = self.Lblock(ner_output)
        return ner_output


# Model
class IC_NER_Model(torch.nn.Module):
    def __init__(self,
                 model_name:str,
                 num_labels:Dict,
                 max_length:int,
                 dim:int=256,
                 dropout:float=.25,
                 device:str='cuda:0',
                 )->None:
        super(IC_NER_Model, self).__init__()
        # Parameters
        self.num_labels = num_labels
        if dim>=max(self.num_labels['IC'],self.num_labels['NER']):
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
        self.hidden_size = config.hidden_size
        self.transformer = AutoModel.from_config(config).to(device)
        # Layers
        self.LFormat = DataFormat(self.hidden_size, self.dim, dropout, device)
        self.ic_layer = NER2IC(self.dim, max_length, self.num_labels['IC'], dropout, device)
        self.ner_layer = IC2NER(self.dim, self.num_labels['NER'], dropout, device)
    
    def forward(self,
                input_ids:torch.Tensor,
                attention_mask:torch.Tensor,
                )->torch.Tensor:
        # Input
        ic_tokens, ner_tokens = self._disentangle_transformer(input_ids, attention_mask)
        ic_tokens, ner_tokens = self.LFormat(ic_tokens, ner_tokens)
        # Info sharing. Reshape NER output as both will be concatenated to be 
        # a single output
        ner_output = self.ner_layer(ic_tokens, ner_tokens)
        ic_output = self.ic_layer(ic_tokens, ner_tokens)
        # Output
        return (ic_output, ner_output)
    
    def _disentangle_transformer(self,
                                 input_ids:torch.Tensor,
                                 attention_mask:torch.Tensor,
                                 )->torch.Tensor:
        # Get HuggingFace output
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Check that output has the desired attribute
        if not hasattr(output, 'last_hidden_state'):
            raise AttributeError(f"Transformers output does not have the attribute 'last_hidden_state'. Please check imported backbone.")
        # Return ic_tokens and ner_tokens
        return torch.mean(output.last_hidden_state, dim=1), output.last_hidden_state