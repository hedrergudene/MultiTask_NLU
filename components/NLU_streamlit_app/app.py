#
# Requirements
#

import streamlit as st
from annotated_text import annotated_text
import json
import torch
from transformers import AutoTokenizer
from model import IC_NER_Model
import fire

#
# Part I: Auxiliary methods
#

# Load model method
def setup_model():
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Load dictionaries
    with open('intent2idx.json', 'r') as fint:
        intent2idx = json.load(fint)
    with open('ner2idx.json', 'r') as fner:
        ner2idx = json.load(fner)
    # Model
    model = IC_NER_Model(
        model_name = 'xlm-roberta-base',
        max_length=30,
        num_labels={'IC': 60, 'NER': 111},
        dim=256,
        dropout=.25,
        device=device,
        )
    model.load_state_dict(torch.load('best_checkpoint.bin', map_location=torch.device(device)))
    for param in model.parameters():
        param.requires_grad = False
    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    return model, tokenizer, intent2idx, ner2idx

# Tokenize input text
def tokenize_text(tokenizer,
                  query:str,
                  ):
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Preprocess labels
    tokens = tokenizer.encode_plus(query,
                                   max_length=30,
                                   padding='max_length',
                                   truncation=True,
                                   return_offsets_mapping=True,
                                   return_tensors='pt',
                                   )
    return {k:v.to(device) for k,v in tokens.items()}

# Annotate entities in text
def annotate_entities(query, ner_label, offset_mapping):
    annot_list = []
    for label, elem in zip(ner_label, offset_mapping):
        if elem[1].item()==0:
            continue
        annot_list.append((query[elem[0]:elem[1]], label))
    return annot_list

# Truncate numbers 
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
    
# Make predictions
def predict(model,
            tokens,
            intent2idx,
            ner2idx,
            query:str,
            threshold:float=.6,
            ):
    # Load model
    model.eval()
    # Get predictions
    idx2intent = {v:k for k,v in intent2idx.items()}
    idx2ner = {v:k for k,v in ner2idx.items()}
    with torch.no_grad():
        outputs = model(tokens['input_ids'], tokens['attention_mask']).detach().cpu()
    ic_logits = outputs[:,:len(intent2idx)]
    ner_logits = outputs[:,len(intent2idx):].reshape((-1, 30, len(ner2idx)))
    ic_label = idx2intent[torch.nn.functional.softmax(ic_logits, dim=-1).argmax().item()]
    ic_conf = torch.nn.functional.softmax(ic_logits, dim=-1).max().item()
    ner_max = torch.nn.functional.softmax(ner_logits, dim=-1).max(dim=-1)
    ner_label = [idx2ner[elem.item()] if prob.item()>threshold else 'O' for prob, elem in zip(ner_max.values.squeeze(),ner_max.indices.squeeze())]
    ner_annotations = annotate_entities(query, ner_label, tokens['offset_mapping'].squeeze().detach().cpu())
    return ic_label, truncate(ic_conf, 3), ner_annotations


#
# Part II: Main code
#

def main():
    st.title('MultiTask NLU Virtual Assistant')
    st.markdown("Custom implementation of MultiTask architecture trained in MASSIVE dataset (Alexa)")
    model, tokenizer, intent2idx, ner2idx = setup_model()
    with st.sidebar:
        query = st.text_input("Write your query", "")
        threshold = st.slider('Choose confidence threshold for entity recognition', 0., 1., .6, .01)
        if query is not None:
            tokens = tokenize_text(tokenizer, query)

        class_btn = st.button("Analyse")
    if class_btn:
        if (query is None) or (threshold is None):
            st.write("Invalid command, please formulate your query")
        else:
            with st.spinner('Model working....'):
                ic_label, ic_conf, ner_annotations = predict(model, tokens, intent2idx, ner2idx, query, threshold)
                st.success('Classified')
            st.write(f"Predicted intent is {ic_label} with confidence {ic_conf}. A detailed labelling is provided now:")
            annotated_text(*ner_annotations)


if __name__=="__main__":
    fire.Fire(main)