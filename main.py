# Requierments
import logging as log
import json
import os
import sys
import torch
import wandb
import fire
import numpy as np
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

# Dependencies
from src.setup import setup_data
from src.dataset import IC_NER_Dataset
from src.model import IC_NER_Model
from src.fitter import CustomTrainer

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

#
# Part I: Read configuration files
#
#Training
with open(os.path.join('input','training_config.json')) as f:
    train_dct = json.load(f)
#Wandb
with open(os.path.join('input','wandb_config.json')) as f:
    wandb_dct = json.load(f)
    os.environ['WANDB_API_KEY'] = wandb_dct['WB_KEY']
    os.environ['WANDB_USERNAME'] = wandb_dct['WB_ENTITY']
    os.environ['WANDB_PROJECT'] = wandb_dct['WB_PROJECT']

#
# Part II: Setup data and model
#

# Get tools
log.info(f"Setup tools:")
data, nlp, intent2idx, ner2idx, max_length, num_labels = setup_data(train_dct)
train_dct['max_length'] = max_length
# Filter data
data['intent_lang'] = data['intent'] + '_' + data['language']
train_text, val_text, train_intent, val_intent = train_test_split(data['utt'].values, data['intent'].values, test_size=.2, stratify=data['intent_lang'])
# Build DataLoaders
log.info(f"Prepare datasets:")
train_dts = IC_NER_Dataset(train_text, train_intent, train_dct['HuggingFace_model'], max_length, nlp, intent2idx, ner2idx)
val_dts = IC_NER_Dataset(val_text, val_intent, train_dct['HuggingFace_model'], max_length, nlp, intent2idx, ner2idx)
# Define model
log.info(f"Get model:")
model = IC_NER_Model(train_dct['HuggingFace_model'], train_dct['max_length'], num_labels, train_dct['dim'], train_dct['dropout'], train_dct['device'])

#
# Part III: Prepare Trainer
#

# Set up arguments
steps_per_epoch = int(.8*len(data))/(train_dct['batch_size']*train_dct['gradient_accumulation_steps'])
logging_steps = steps_per_epoch if int(steps_per_epoch)==steps_per_epoch else int(steps_per_epoch)+1
logging_steps = logging_steps//4
# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(os.getcwd(),train_dct['filepath']),
    gradient_accumulation_steps=train_dct['gradient_accumulation_steps'],
    warmup_steps=logging_steps,
    learning_rate=train_dct['learning_rate'],
    weight_decay=train_dct['weight_decay'],
    per_device_train_batch_size=train_dct['batch_size'],
    per_device_eval_batch_size=train_dct['batch_size'],
    dataloader_num_workers = train_dct['dataloader_num_workers'],
    num_train_epochs=train_dct['epochs'],
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=logging_steps,
    logging_strategy="steps",
    logging_steps=logging_steps,
    report_to="wandb",  # enable logging to W&B
    run_name=wandb_dct['WB_RUN_NAME'],
    seed=train_dct['seed'],
    fp16=bool(train_dct['fp16'])
)
# Trainer
trainer = CustomTrainer(
    model,
    training_args,
    train_dataset=train_dts,
    eval_dataset=val_dts,
)

#
# Part IV: Train model
#

# Trainer
trainer.train()

#
# Part V: Evaluation (WIP)
#

# End WB session
wandb.finish()
