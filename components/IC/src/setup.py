# Requirements
import os
import requests
import tarfile
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
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
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(f, "input/")
    os.remove('input/amazon-massive-dataset-1.0.tar.gz')
    # Create a dicitonary to gather all entities
    utterances = []
    intents = []
    lang = []
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
    # Load and gather data
    data = pd.DataFrame({'utt':utterances, 'intent':intents, 'language':lang})

    #
    # Part II: Tokenizer and data insights
    #

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_dct['HuggingFace_model'])
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
    length_list = []
    for elem in tqdm(utterances):
        length_list.append(len(tokenizer(elem).input_ids))
    max_length = int(np.quantile(length_list, .995))
    log.info(f"Recommended maximum length: {max_length}")

    # Exit
    return data, intent2idx, max_length, num_labels
