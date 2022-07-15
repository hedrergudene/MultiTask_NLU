
# Requirements
import os
import requests
import itertools
import tarfile
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import AutoConfig, XLMRobertaTokenizerFast
import logging as log
from typing import Dict

# Method to load data and compute 
def setup_data(train_dct:Dict,
               url:str='https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz',
               ):
    #
    # Part I: Data gathering
    #

    # Generate input directory and download data
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open('input/amazon-massive-dataset-1.0.tar.gz', 'wb') as f:
            f.write(response.raw.read())
    # Extract files in input directory
    with tarfile.open('input/amazon-massive-dataset-1.0.tar.gz') as f:
      f.extractall('input/')
    os.remove('input/amazon-massive-dataset-1.0.tar.gz')
    # Create a dicitonary to gather all entities
    utterances = []
    intents = []
    lang = []
    ent_dct = {}
    # Loop through all languages
    for elem in [file_id for file_id in os.listdir('input/1.0/data') if file_id[:2]!='._']:
        # Open file
        with open(os.path.join('input/1.0/data', elem), 'r') as json_file:
            json_list = list(json_file)
        log.info(f"Processing language: {elem.split('.')[0]}")
        # Iterate through all annotations
        for json_str in json_list:
            result = json.loads(json_str)
            utterances.append(result['utt'])
            intents.append(result['intent'])
            lang.append(elem.split('.')[0])
            # Iterate through all possible entities
            str_sample = result['annot_utt']
            while str_sample.find('[')!=-1 and str_sample.find(']')!=-1:
                # Get (key, value) pair
                [key, value] = str_sample[str_sample.find('[')+1:str_sample.find(']')].split(' : ')
                # Include in dictionary if it's necessary
                if key not in ent_dct.keys():
                    ent_dct[key] = [value]
                elif key in ent_dct.keys() and value not in ent_dct.values():
                    ent_dct[key].append(value)
                # Update string
                str_sample = str_sample[str_sample.find(']')+1:]
    # Load and gather data
    data = pd.DataFrame({'utt':utterances, 'intent':intents, 'language':lang})

    #
    # Part II: Tokenizer and data insights
    #

    # Tokenizer
    teacher_tokenizer = XLMRobertaTokenizerFast.from_pretrained(train_dct['teacher_model'])
    student_tokenizer = XLMRobertaTokenizerFast.from_pretrained(train_dct['student_model'])
    # Data insights
    ## IC
    intent2idx = {k:v for k,v in zip(data['intent'].sort_values().unique(), range(data['intent'].nunique()))}
    idx2intent = {v:k for k,v in intent2idx.items()}
    ## Num labels
    num_labels = len(intent2idx)


    #
    # Part III: Max length
    #

    # Estimate max length
    teacher_length_list = []
    student_length_list = []
    for elem in tqdm(utterances):
        teacher_length_list.append(len(teacher_tokenizer(elem).input_ids))
        student_length_list.append(len(student_tokenizer(elem).input_ids))
    max_length = {'teacher':int(np.quantile(teacher_length_list, .995)), 'student':int(np.quantile(student_length_list, .995))}
    log.info(f"Recommended maximum length: {max_length}")

    # Exit
    return data, intent2idx, max_length, num_labels