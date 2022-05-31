# Requirements
import requests
import itertools
import tarfile
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import AutoConfig, AutoTokenizer
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
        print(f"Processing language: {elem.split('.')[0]}")
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
    tokenizer = AutoTokenizer.from_pretrained(train_dct['HuggingFace_model'])
    # Data insights
    ## IC
    intent2idx = {k:v for k,v in zip(data['intent'].sort_values().unique(), range(data['intent'].nunique()))}
    idx2intent = {v:k for k,v in intent2idx.items()}
    ## NER
    num_tags = num_tags = len(list(train_dct["scheme"]))*len(ent_dct.keys())
    ner2idx = {(str(elem[0])+"-"+str(elem[1])):i for i,elem in zip(range(1,num_tags+1), itertools.product(list(train_dct["scheme"]),list(ent_dct.keys())))}
    ner2idx['O'] = 0 # Add outside tag
    # UPDATE: REMOVE -100 TOKEN
    #ner2idx['PAD'] = -100 # Add padding tag
    idx2ner = {v:k for k,v in ner2idx.items()}
    ## Num labels
    num_labels = {'IC':len(intent2idx), 'NER':len(ner2idx)}


    #
    # Part III: Create spaCy entity rule objects
    #

    # Create spaCy model and add Entity Ruler
    ents = [[key for _ in values]for key, values in ent_dct.items()]
    flat_ents = []
    for elem in ents:
      flat_ents = flat_ents + elem
    flat_patterns = []
    for _, v in ent_dct.items():
        flat_patterns = flat_patterns + v
    entities = pd.DataFrame({'pattern':flat_patterns, 'label':flat_ents})
    patterns = entities[["pattern", 'label']].dropna().drop_duplicates()
    # Create spaCy model and add Entity Ruler
    nlp = spacy.load(train_dct["spaCy_model"], exclude=["ner"])
    config = {
        "phrase_matcher_attr": None,
        "validate": False, #	Whether patterns should be validated (passed to the Matcher and PhraseMatcher)
        "overwrite_ents": True, # If existing entities are present, e.g. entities added by the model, overwrite them by matches if necessary.
        "ent_id_sep": "||", # Separator used internally for entity IDs
    }
    ruler = nlp.add_pipe("entity_ruler", config=config)
    assert len(ruler) == 0
    ruler.add_patterns(list(patterns.T.to_dict().values()))
    assert len(ruler) == len(patterns)
    log.info(f'Number of patterns added: {len(ruler)}. Entities associated to those patterns: {entities.nunique()}')

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
    return data, nlp, intent2idx, ner2idx, max_length, num_labels
