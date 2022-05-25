# Requirements
import requests
import itertools
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import AutoConfig, AutoTokenizer
import logging as log

# Method to load data and compute 
def setup_data(setup_config_path:str='input/setup_config.json'):
    #
    # PART I: Data gathering
    #

    # Load parameters
    with open(setup_config_path, 'r') as f:
        setup_config = json.load(f)

    # Train data fetch
    response = requests.get("https://raw.githubusercontent.com/howl-anderson/ATIS_dataset/master/data/standard_format/rasa/train.json")
    train_data = response.json()['rasa_nlu_data']['common_examples']
    # Test data fetch
    response = requests.get("https://raw.githubusercontent.com/howl-anderson/ATIS_dataset/master/data/standard_format/rasa/test.json")
    test_data = response.json()['rasa_nlu_data']['common_examples']
    # Fetch entity labels and associated expressions
    texts = []
    intents = []
    ent_label = []
    ent_pattern = []
    i = 0

    for dict_data in train_data:
        texts.append(dict_data["text"])
        intents.append(dict_data["intent"])
        i += 1
        for dict_ents in dict_data["entities"]:
            ent_label.append(dict_ents['entity'])
            ent_pattern.append(dict_ents['value'])
    
    for dict_data in test_data:
        texts.append(dict_data["text"])
        intents.append(dict_data["intent"])
        for dict_ents in dict_data["entities"]:
            ent_label.append(dict_ents['entity'])
            ent_pattern.append(dict_ents['value'])
    
    entities = pd.DataFrame({"label":ent_label, "pattern":ent_pattern}).drop_duplicates().reset_index(drop=True)

    #
    # PART II: Multilabel IC data
    #
    # Binarise intent labels
    unique_intents = [elem for elem in list(set(intents)) if "+" not in list(elem)]
    intent2idx = {elem:i for i,elem in zip(range(len(unique_intents)), unique_intents)}

    #
    # Part III: Create spaCy entity rule objects
    #
    
    # PART I: Create spaCy model and add Entity Ruler
    patterns = entities[["pattern", 'label']].dropna().drop_duplicates()
    # PART II: Create spaCy model and add Entity Ruler
    nlp = spacy.load(setup_config["spaCy_model"], exclude=["ner"])
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
    # Part IV: Model settings
    #
    # Estimate max length
    tokenizer = AutoTokenizer.from_pretrained(setup_config["HuggingFace_model"])
    length_list = []
    for elem in tqdm(utterances):
        length_list.append(len(tokenizer(elem).input_ids))
    max_length = np.quantile(length_list, .995)
    log.info(f"Recommended maximum length: {max_length}")
    # Tag systems in NER
    labels = entities['label'].sort_values().unique()
    num_tags = len(list(setup_config["scheme"]))*len(labels)
    ner2idx = {(str(elem[0])+"-"+str(elem[1])):i for i,elem in zip(range(1,num_tags+1), itertools.product(list(setup_config["scheme"]),labels))}
    ner2idx['O'] = 0 # Add outside tag
    ner2idx['PAD'] = -100 # Add padding tagç
    
    # Build original tag2idx dictionary for metrics purposes
    labels = pd.Series(ent_label).sort_values().unique()
    num_tags = len(list(setup_config["scheme"]))*len(labels)
    # Number of classes of each problem
    num_labels = {'IC':len(intent2idx),
                  'NER':len(ner2idx)-1, # Remove 'PAD' from classification
                  }
    # Exit
    return {'train':texts[:i], 'test':texts[i:]}, {'train':intents[:i], 'test':intents[i:]}, nlp, intent2idx, ner2idx, max_length, num_labels
