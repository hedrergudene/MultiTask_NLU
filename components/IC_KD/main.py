
# Requierments
import logging as log
import json
import requests
import os
import shutil
import sys
import torch
import wandb
import fire
import numpy as np
import pandas as pd
import plotly_express as px
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Dependencies
from src.setup import setup_data
from src.dataset import KDDataset
from src.loss import DistilLoss
from src.model import IC_Model, KD_IC_Model
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
    with open(training_config) as f:
        train_dct = json.load(f)
        if train_dct['teacher_checkpoint'].split('.')[-1]=='bin':
            pass
        else:
            with requests.get(train_dct['teacher_checkpoint'], allow_redirects=True) as req:
                open('input/best_teacher_checkpoint.bin', 'wb').write(req.content)
                train_dct['teacher_checkpoint'] = 'input/best_teacher_checkpoint.bin'
        seed_everything(train_dct['seed'])
    #Wandb
    with open(wandb_config) as f:
        wandb_dct = json.load(f)
        os.environ['WANDB_API_KEY'] = wandb_dct['WB_KEY']
        os.environ['WANDB_USERNAME'] = wandb_dct['WB_ENTITY']
        os.environ['WANDB_PROJECT'] = wandb_dct['WB_PROJECT']

    #
    # Part II: Setup data and model
    #

    # Get tools
    print(f"Setup tools:")
    data, intent2idx, max_length, num_labels = setup_data(train_dct)
    # Filter data
    data['intent_lang'] = data['intent'] + '_' + data['language']
    train_idx, val_idx, train_intent, val_intent = train_test_split(np.arange(0,len(data)), data['intent'].values, test_size=.2, stratify=data['intent_lang'])
    # Build datasets
    log.info(f"Prepare datasets:")
    train_dts = KDDataset(data.loc[train_idx,'utt'].values, train_intent, train_dct['teacher_model'], train_dct['student_model'], max_length, intent2idx)
    val_dts = KDDataset(data.loc[val_idx,'utt'].values, val_intent, train_dct['teacher_model'], train_dct['student_model'], max_length, intent2idx)
    # Define model
    print(f"Get model:")
    model = KD_IC_Model(train_dct['student_model'], max_length['student'], num_labels, train_dct['dropout'], train_dct['device'])

    #
    # Part III: Prepare Trainer
    #

    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Set up arguments
    print(f"Prepare training arguments:")
    steps_per_epoch = int(.8*len(data))/(train_dct['batch_size']*train_dct['gradient_accumulation_steps'])
    logging_steps = steps_per_epoch if int(steps_per_epoch)==steps_per_epoch else int(steps_per_epoch)+1
    logging_steps = logging_steps//train_dct['evaluation_steps_per_epoch']
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(os.getcwd(),train_dct['filepath']),
        gradient_accumulation_steps=train_dct['gradient_accumulation_steps'],
        warmup_steps=logging_steps*train_dct['warmup_steps_factor'],
        learning_rate=train_dct['learning_rate'],
        weight_decay=train_dct['weight_decay'],
        per_device_train_batch_size=train_dct['batch_size'],
        per_device_eval_batch_size=2*train_dct['batch_size'],
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

    teacher_model = IC_Model(train_dct['teacher_model'], max_length['teacher'], num_labels, train_dct['dropout'], train_dct['device'])
    distil_loss_ic = DistilLoss(teacher_model=teacher_model, teacher_checkpoint = train_dct['teacher_checkpoint'], alpha = train_dct['alpha_distil'], n_classes = num_labels)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            ic_labels = torch.squeeze(inputs.get('labels'))
            # forward pass
            outputs = model(inputs.get('input_ids'), inputs.get('attention_mask'))
            # compute custom loss
            ic_loss = distil_loss_ic({'input_ids':inputs.get('teacher_input_ids'), 'attention_mask':inputs.get('teacher_attention_mask')},
                                     outputs,
                                     ic_labels,
                                     )
            return (ic_loss, outputs) if return_outputs else ic_loss

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
    lang_df = pd.DataFrame(lang_dct).reset_index().melt(id_vars='index')
    
    fig_lang_IC = px.bar(lang_df.loc[lang_df['index']=='f1_IC',:],
                         x="variable",
                         y="value",
                         color="variable",
                         title="Intent classification f1-score per language",
                         ).update_xaxes(categoryorder='total descending')

    wandb.log({"Intent classification f1-score per language": fig_lang_IC})
    wandb.log({'Global metrics':wandb.Table(data=[list(metrics_dct.values())], columns=list(metrics_dct.keys()))})

    # End WB session
    print(f"End Weights and Biases session:")
    last_model=max([int(elem.split('-')[-1]) for elem in os.listdir(os.path.join(os.getcwd(),'output'))])
    shutil.move(os.path.join(os.getcwd(),'output',f"checkpoint-{last_model}"), os.path.join(f"{wandb.run.dir}",f"checkpoint-{last_model}"))
    wandb.finish()
   

if __name__=="__main__":
    fire.Fire(main)