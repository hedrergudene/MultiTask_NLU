# Requierments
import logging as log
import json
import os
import shutil
import sys
import torch
import wandb
import fire
import numpy as np
import pandas as pd
import plotly_express as px
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

# Dependencies
from src.setup import setup_data
from src.dataset import IC_NER_Dataset
from src.model import IC_NER_Model
from src.fitter import CustomTrainer
from src.metrics import evaluate_metrics
from src.utils import seed_everything

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
        training_config:str="input/training_config.json",
        wandb_config:str="input/wandb_config.json",
        ):

    #
    # Part I: Read configuration files
    #
    
    #Training
    with open(os.path.join('input','training_config.json')) as f:
        train_dct = json.load(f)
        seed_everything(train_dct['seed'])
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
    print(f"Setup tools:")
    data, nlp, intent2idx, ner2idx, max_length, num_labels = setup_data(train_dct)
    train_dct['max_length'] = max_length
    # Filter data
    data['intent_lang'] = data['intent'] + '_' + data['language']
    train_idx, val_idx, train_intent, val_intent = train_test_split(np.arange(0,len(data)), data['intent'].values, test_size=.2, stratify=data['intent_lang'])
    # Build datasets
    log.info(f"Prepare datasets:")
    train_dts = IC_NER_Dataset(data.loc[train_idx,'utt'].values, train_intent, train_dct['HuggingFace_model'], max_length, nlp, intent2idx, ner2idx)
    val_dts = IC_NER_Dataset(data.loc[val_idx,'utt'].values, val_intent, train_dct['HuggingFace_model'], max_length, nlp, intent2idx, ner2idx)
    # Define model
    print(f"Get model:")
    model = IC_NER_Model(train_dct['HuggingFace_model'], train_dct['max_length'], num_labels, train_dct['dim'], train_dct['dropout'], train_dct['device'])

    #
    # Part III: Prepare Trainer
    #
    
    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Set up arguments
    print(f"Prepare training arguments:")
    steps_per_epoch = int(.8*len(data))/(train_dct['batch_size']*train_dct['gradient_accumulation_steps'])
    logging_steps = steps_per_epoch if int(steps_per_epoch)==steps_per_epoch else int(steps_per_epoch)+1
    logging_steps = logging_steps//4
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(os.getcwd(),train_dct['filepath']),
        gradient_accumulation_steps=train_dct['gradient_accumulation_steps'],
        warmup_steps=logging_steps*train_dct['warmup_steps_factor'],
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
    print(f"Initialise HuggingFace Trainer:")
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
    print(f"Start model training:")
    trainer.train()

    #
    # Part V: Evaluation (WIP)
    #

    # Prepare evaluation DataLoader and language list
    val_dtl = torch.utils.data.DataLoader(val_dts,
                                          batch_size=trainer.args.per_device_eval_batch_size,
                                          num_workers=trainer.args.dataloader_num_workers,
                                          shuffle=False, # Important to be aligned with lang list
                                          )
    language_arr = data.loc[val_idx, 'language'].values

    # Calculate metrics
    print("Compute metrics on evaluation dataset:")
    metrics_dct, lang_dct = evaluate_metrics(trainer, val_dtl, language_arr)

    # Log metrics
    lang_df = pd.DataFrame(lang_metrics).reset_index().melt(id_vars='index')
    
    fig_lang_IC = px.bar(lang_df.loc[lang_df['index']=='f1_IC',:],
                         x="variable",
                         y="value",
                         color="variable",
                         title="Intent classification f1-score per language",
                         ).update_xaxes(categoryorder='total descending')

    fig_lang_NER = px.bar(lang_df.loc[lang_df['index']=='f1_NER',:],
                          x="variable",
                          y="value",
                          color="variable",
                          title="Entity recognition f1-score per language",
                          ).update_xaxes(categoryorder='total descending')

    wandb.log({"Intent classification f1-score per language": fig_lang_IC})
    wandb.log({"Entity recognition f1-score per language": fig_lang_NER})
    wandb.log({'global_'+k:v for k,v in global_metrics.items()})

    # End WB session
    print(f"End Weights and Biases session:")
    last_model=max([int(elem.split('-')[-1]) for elem in os.listdir(os.path.join(os.getcwd(),'output'))])
    shutil.move(os.path.join(os.getcwd(),'output',f"checkpoint-{last_model}"), os.path.join(f"{wandb.run.dir}",f"checkpoint-{last_model}"))
    wandb.finish()
   

if __name__=="__main__":
    fire.Fire(main)
