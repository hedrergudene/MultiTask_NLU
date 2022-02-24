# Requirements
import requests
import itertools
import numpy as np
import pandas as pd
import spacy
from transformers import AutoConfig

# Method to load data and compute 
def setup_data(HuggingFace_model:str='roberta-base',
               spaCy_model:str='en_core_web_sm',
               scheme:str='IB',
               ):
    #
    # PART I: Data gathering
    #

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

    for dict_data in train_data:
        texts.append(dict_data["text"])
        intents.append(dict_data["intent"])
        for dict_ents in dict_data["entities"]:
            ent_label.append(dict_ents['entity'])
            ent_pattern.append(dict_ents['value'])
    
    entities = pd.DataFrame({"label":ent_label, "pattern":ent_pattern}).drop_duplicates().reset_index(drop=True)
    entities["label_0"] = entities["label"].apply(lambda x: x.split(".")[0])
    entities["label_1"] = entities["label"].apply(lambda x: x.split(".")[1] if len(x.split("."))>1 else np.NaN)
    entities = entities.drop(["label"], axis=1)
    entities = entities[["label_0", "label_1", "pattern"]]

    #
    # PART II: Multilabel IC data
    #
    # Auxiliary method
    def multilabel_intent(intent2idx, labels):
        target = np.zeros((1,len(intent2idx)))
        for label in labels:
            target[0,intent2idx[label]] = 1
        return target
    # Binarise intent labels
    unique_intents = [elem for elem in list(set(intents)) if "+" not in list(elem)]
    intent2idx = {elem:i for i,elem in zip(range(len(unique_intents)), unique_intents)}
    multilabel_intents = np.concatenate([multilabel_intent(intent2idx,elem.split("+")) for elem in intents], axis = 0)

    #
    # Part III: Create spaCy entity rule objects
    #
    # Object to store entity rulers
    tag2nlp={}
    tags = [col for col in entities.columns if col[:5]=="label"]
    # Loop
    for column in tags:
        # PART I: Create spaCy model and add Entity Ruler
        patterns = entities[["pattern", column]].dropna().drop_duplicates()
        patterns.columns = ["pattern", "label"]
        # PART II: Create spaCy model and add Entity Ruler
        nlp = spacy.load(spaCy_model, exclude=["ner"])
        config = {
           "phrase_matcher_attr": None,
           "validate": False, #	Whether patterns should be validated (passed to the Matcher and PhraseMatcher)
           "overwrite_ents": True, # If existing entities are present, e.g. entities added by the model, overwrite them by matches if necessary.
           "ent_id_sep": "||", # Separator used internally for entity IDs
        }
        # ~5 min
        ruler = nlp.add_pipe("entity_ruler", config=config)
        assert len(ruler) == 0
        ruler.add_patterns(list(patterns.T.to_dict().values()))
        assert len(ruler) == len(patterns)
        print(f'Number of patterns added in level {column}: {len(ruler)}. Entities associated to those patterns: {entities[column].nunique()}')
        # PART III: Save results
        tag2nlp[column]=nlp
    
    #
    # Part IV: Model settings
    #
    # Set model parameters
    config = AutoConfig.from_pretrained(HuggingFace_model)
    MAX_LEN = config.max_position_embeddings
    tag2idxs={}
    # Tag systems in NER
    for column in tags:
        labels = entities[column].sort_values().unique()
        num_tags = len(list(scheme))*len(labels)
        # Create dictionary for each tag type
        idx2tag = {i:(str(elem[0])+"-"+str(elem[1])) for i,elem in zip(range(1,num_tags+1), itertools.product(list(scheme),labels))}
        idx2tag[0] = 'O' # Add outside tag
        idx2tag[-100] = 'PAD' # Add padding tag
        tag2idx = {v:k for k,v in idx2tag.items()}
        tag2idxs[column] = tag2idx
    
    # Build reverse ones
    idxs2tag = {column: {v:k for k,v in tag2idxs[column].items()} for column in tags}
    # Build original tag2idx dictionary for metrics purposes
    labels = pd.Series(ent_label).sort_values().unique()
    num_tags = len(list(scheme))*len(labels)
    original_idxs2tag = {i:(str(elem[0])+"-"+str(elem[1])) for i,elem in zip(range(1,num_tags+1), itertools.product(list(scheme),labels))}
    original_idxs2tag[0] = 'O' # Add outside tag
    original_idxs2tag[-100] = 'PAD' # Add padding tag
    # Number of classes of each problem
    num_labels = {'IC':len(unique_intents),
                  'H_NER':{column:len(tag2idxs[column])-1 for column in tags},
                  }
    # Exit
    return texts, multilabel_intents, tags, tag2nlp, tag2idxs, idxs2tag, original_idxs2tag, MAX_LEN, num_labels
