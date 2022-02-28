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
        self.block = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim, device=device),
                                          torch.nn.LayerNorm(output_dim, device=device),
                                          torch.nn.GELU(),
                                          torch.nn.Dropout(dropout),
                                        )
    
    def forward(self, x):
        return self.block(x)


## Attn product
def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
    output, attention_weights
    """
    # Matrix multiplication
    matmul_qk = torch.matmul(q, k.transpose(1,2))  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


## Data formatting and projection
class LinearProjFormat(torch.nn.Module):
    def __init__(self,
                 tags:List,
                 hidden_size:int,
                 proj_dim:int=128,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(LinearProjFormat, self).__init__()
        # Parameters
        self.linear = {'IC':LinearBlock(hidden_size, proj_dim, dropout=dropout, device=device),
                        'H_NER':{column:LinearBlock(hidden_size, proj_dim, dropout=dropout, device=device) for column in tags},
                        }
        self.tags = tags
    
    def forward(self, pooler_output, sequence_output):
        formatted_output = {"IC":self.linear['IC'](pooler_output),
                            "H_NER":{column:self.linear['H_NER'][column](sequence_output) for column in self.tags},
                            }
        return formatted_output


## Information sharing from IC to NER
class IC2NER(torch.nn.Module):
    def __init__(self,
                 tags:List,
                 num_labels:Dict,
                 proj_dim:int=128,
                 num_heads:int=4,
                 dropout:float=.25,
                 device='cuda:0',
                 ):
        super(IC2NER, self).__init__()
        # Parameters
        self.tags = tags
        # IC reshape
        self.embedding = LinearBlock(1, proj_dim, dropout, device)
        # H_NER multi-head attn products
        self.attn = {column:torch.nn.MultiheadAttention(proj_dim, num_heads, kdim=proj_dim, vdim=proj_dim, batch_first=True, device=device) for column in self.tags[:-1]}
        # H_NER reshape
        self.linear = {column:LinearBlock(proj_dim, num_labels['H_NER'][column], dropout=dropout, device=device) for column in self.tags[:-1]}
    
    def forward(self, formatted_tensor):
        # IC branch setup
        input = formatted_tensor.copy()
        input['IC'] = self.embedding(torch.unsqueeze(input['IC'], dim=-1)) # Shape (batch_size, proj_dim, proj_dim)
        # H_NER tensors
        for i in range(len(self.tags)-1):
            input['H_NER'][self.tags[i+1]], _ = self.attn[self.tags[i]](query=input['H_NER'][self.tags[i]],
                                                                        key=input['H_NER'][self.tags[i+1]],
                                                                        value=input['H_NER'][self.tags[i+1]],
                                                                        )
        # Mix
        ner_output = {column: self.linear[column](torch.bmm(input['H_NER'][column], input['IC'])) for column in self.tags}
        return ner_output


## Information sharing from NER to IC
class NER2IC(torch.nn.Module):
    def __init__(self,
                 num_labels:Dict,
                 proj_dim:int=128,
                 dropout:float=.25,
                 device:str='cuda',
                 ):
        super(NER2IC, self).__init__()
        # Parameters
        self.linear = LinearBlock(proj_dim, num_labels['IC'], dropout = dropout, device=device)
        self.weights_ner = torch.nn.Parameter(torch.zeros((len(num_labels['H_NER'])))).to(device)
    
    def forward(self, ic_tensor, ner_output):
        ic_tensor = torch.unsqueeze(ic_tensor, dim=-1) # Shape (batch_size, proj_dim, 1)
        w = torch.sigmoid(self.weights_ner)/torch.sum(torch.sigmoid(self.weights_ner))
        ner_tensor = torch.mean(w[0]*ner_output['label_0']+w[1]*ner_output['label_1'], dim=1, keepdim=True) # Shape (batch_size, 1, proj_dim)
        output = torch.matmul(ic_tensor, ner_tensor) # Shape (batch_size, proj_dim, proj_dim)
        output = torch.mean(output, dim=-1, keepdim=False) # Shape (batch_size, proj_dim)
        output = self.linear(output) # Shape (batch_size, num_classes['IC'])
        return output


# Model
class MT_IC_HNER_Model(torch.nn.Module):
    def __init__(self,
                 model_name:str,
                 num_labels:Dict,
                 proj_dim:int=128,
                 num_heads:int=4,
                 hidden_dropout_prob:float=.1,
                 layer_norm_eps:float=1e-7,
                 dropout:float=.25,
                 device:str='cuda:0',
                 ):
        super(MT_IC_HNER_Model, self).__init__()
        # Parameters
        self.num_heads = num_heads
        self.num_labels = num_labels
        self.proj_dim = proj_dim
        self.tags = list(self.num_labels["H_NER"].keys())
        # Take pretrained model from custom configuration
        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        self.pos_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.transformer = AutoModel.from_config(config).to(device)
        # Format
        self.format_layer = LinearProjFormat(self.tags, self.hidden_size, self.proj_dim, dropout= dropout, device=device)
        # NER manipulation
        self.ic2ner = IC2NER(self.tags, self.num_labels, self.proj_dim, self.num_heads, dropout= dropout, device=device)
        # IC ensemble
        self.ner2ic = NER2IC(self.num_labels, self.proj_dim, dropout= dropout, device=device)
    
    def forward(self, tokens, attn_mask):
        transformer_output = self.transformer(tokens, attn_mask)
        pooler_output = transformer_output.pooler_output # Shape (batch, hidden_size)
        sequence_output = transformer_output.last_hidden_state # Shape (batch, max_position_embeddings, hidden_size)
        formatted_dict = self.format_layer(pooler_output, sequence_output)
        ner_output = self.ic2ner(formatted_dict)
        ic_output = self.ner2ic(formatted_dict['IC'], ner_output)
        return {'IC':ic_output,
                'H_NER':ner_output,
                }