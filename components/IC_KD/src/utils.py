
# Requirements
import numpy as np
import torch
import random
import os

# Set seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Method to ensemble NER outputs into one
def convert_tags(tag, original_idxs):
    tag1, tag2= tag.split('.')[0], tag.split('.')[1]
    # EXCLUSIVE CONDITIONS
    ## Check if the first of them is 'O'
    if tag1=='O':
        return 'O'
    ## Both will agree on pads so does not matter which to choose
    elif (tag1=='PAD') or (tag2=='PAD'):
        return 'PAD'
    ## Check if both agree in the same scheme word, otherwise return 'O'
    ## (first condition is redundant given previous ones but we keep for clarity)
    elif (tag1[0] in ['I','B']) and (tag2[0] in ['I','B']) and (tag1[0]!=tag2[0]):
        return 'O'
    ## At this point, first tag is something 'relevant' and alligned with the 
    ## second one, possibly being the latter 'O'. So now we start with...
    
    # INCLUSIVE CONDITIONS
    ## Check if second one is 'O' and first one is in original dict
    elif (tag2=='O') and (tag1 in list(original_idxs.values())):
        return tag1
    ## Check if second one is not 'O' and smart concatenation of them is in
    ## original dictionary; i.e., remove scheme from second (they already agree)
    elif (tag2!='O') and (tag1+'.'+tag2[2:] in list(original_idxs.values())):
        return tag1+tag2[2:]
    
    # ESCAPE CONDITION
    ## The last use case is that the concatenation of them do not lie in the
    ## original dictionary, so we assign 'O'
    else:
        return 'O'