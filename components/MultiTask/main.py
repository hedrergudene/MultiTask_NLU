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
from sklearn.model_selection import train_test_split

# Dependencies
from src.setup import setup_data
from src.dataset import IC_NER_Dataset
from src.model import IC_NER_Model
from src.loss import IC_NER_Loss
from src.fitter import IC_NER_Fitter
from src.callbacks import wandb_checkpoint
from src.metrics import evaluate_metrics
from src.utils import seed_everything

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


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
        seed_everything(train_dct['seed'])
    #Wandb
    with open(wandb_config) as f:
        wandb_dct = json.load(f)
        os.environ['WANDB_API_KEY'] = wandb_dct['WB_KEY']
        os.environ['WANDB_USERNAME'] = wandb_dct['WB_ENTITY']
        os.environ['WANDB_PROJECT'] = wandb_dct['WB_PROJECT']
    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #
    # Part II: Setup data and model
    #

    # Get tools
    log.info(f"Setup tools:")
    data, nlp, intent2idx, ner2idx, max_length, num_labels = setup_data(train_dct)
    train_dct['max_length'] = max_length
    # Filter data
    data['intent_lang'] = data['intent'] + '_' + data['language']
    train_idx, val_idx, train_intent, val_intent = train_test_split(np.arange(0,len(data)), data['intent'].values, test_size=.2, stratify=data['intent_lang'], random_state = train_dct['seed'])
    # Build datasets
    log.info(f"Prepare datasets:")
    train_dts = IC_NER_Dataset(data.loc[train_idx,'utt'].values, train_intent, train_dct['HuggingFace_model'], max_length, nlp, intent2idx, ner2idx)
    train_dtl = torch.utils.data.DataLoader(train_dts,
                                            batch_size=train_dct['batch_size'],
                                            num_workers=train_dct['num_workers'],
                                            shuffle=True,
                                            )
    val_dts = IC_NER_Dataset(data.loc[val_idx,'utt'].values, val_intent, train_dct['HuggingFace_model'], max_length, nlp, intent2idx, ner2idx)
    val_dtl = torch.utils.data.DataLoader(val_dts,
                                          batch_size=2*train_dct['batch_size'],
                                          num_workers=train_dct['num_workers'],
                                          shuffle=False,
                                          )
    # Define model
    log.info(f"Prepare model, loss function, optimizer and scheduler")
    model = IC_NER_Model(train_dct['HuggingFace_model'], num_labels, train_dct['max_length'], train_dct['dim'], train_dct['dropout'], train_dct['device'])
    # Get loss, optimisers and schedulers
    criterion = IC_NER_Loss(loss_type=train_dct['loss_type'],
                            gamma=train_dct['gamma_loss'],
                            temperature=train_dct['temperature_loss'],
                            label_smoothing=train_dct['label_smoothing_loss'],
                            from_logits=True,
                            multilabel=False,
                            reduction='mean',
                            n_classes=num_labels,
                            class_weights=None,
                            device=train_dct['device'],
                            )
    optimizer = torch.optim.AdamW([{'params':model.parameters(), 'lr':train_dct['learning_rate']}, 
                                   {'params':criterion.parameters(), 'lr':train_dct['learning_rate_alpha']}],
                                  lr=train_dct['learning_rate'],
                                  weight_decay=train_dct['weight_decay'],
                                  )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=train_dct['learning_rate'],
                                                    steps_per_epoch=len(train_dtl),
                                                    epochs=train_dct['epochs'],
                                                    pct_start=train_dct['warmup_epochs_factor'],
                                                    anneal_strategy='cos',
                                                    )
    # Fitter
    if not os.path.isdir(os.path.join(os.getcwd(),train_dct['filepath'])): os.makedirs(os.path.join(os.getcwd(),train_dct['filepath']))
    fitter = IC_NER_Fitter(model,
                           train_dct['device'],
                           criterion,
                           optimizer,
                           scheduler,
                           step_scheduler=True,
                           validation_scheduler=False,
                           folder=os.path.join(os.getcwd(),train_dct['filepath']),
                           use_amp = bool(train_dct['use_amp']),
                           )

    # Weights and Biases session
    wandb.login(key=wandb_dct['WB_KEY'])
    wandb.init(project=wandb_dct['WB_PROJECT'], entity=wandb_dct['WB_ENTITY'], config=train_dct)
    # Training
    log.info(f"Start fitter training:")
    _ = fitter.fit(train_loader = train_dtl,
                   val_loader = val_dtl,
                   n_epochs = train_dct['epochs'],
                   metrics = None,
                   early_stopping = train_dct['early_stopping'],
                   early_stopping_mode = train_dct['scheduler_mode'],
                   verbose_steps = train_dct['verbose_steps'],
                   step_callbacks = [wandb_checkpoint],
                   validation_callbacks = [wandb_checkpoint],
                   )
    # Remove objects from memory
    del fitter, criterion, optimizer, scheduler, train_dts, train_dtl
    torch.cuda.empty_cache()

    #
    # Part V: Evaluation
    #

    # Prepare evaluation DataLoader and language list
    language_arr = data.loc[val_idx, 'language'].values

    # Load best checkpoint
    log.info("Loading best model checkpoint:")
    ckpt = torch.load(os.path.join(os.getcwd(),train_dct['filepath'],'best-checkpoint.bin'))
    model = IC_NER_Model(train_dct['HuggingFace_model'], num_labels, train_dct['max_length'], train_dct['dim'], train_dct['dropout'], train_dct['device'])
    model.load_state_dict(ckpt['model_state_dict'])
    # Calculate metrics
    log.info("Compute metrics on evaluation dataset:")
    metrics_dct, lang_dct = evaluate_metrics(model, train_dct, val_dtl, language_arr)

    # Log metrics
    lang_df = pd.DataFrame(lang_dct).reset_index().melt(id_vars='index')
    
    fig_lang_IC = px.bar(lang_df.loc[lang_df['index']=='f1_weighted_IC',:],
                         x="variable",
                         y="value",
                         color="variable",
                         title="Intent classification weighted f1-score per language",
                         ).update_xaxes(categoryorder='total descending')

    fig_lang_NER = px.bar(lang_df.loc[lang_df['index']=='f1_NER',:],
                          x="variable",
                          y="value",
                          color="variable",
                          title="Entity recognition f1-score per language",
                          ).update_xaxes(categoryorder='total descending')

    wandb.log({"Intent classification f1-score per language": fig_lang_IC})
    wandb.log({"Entity recognition f1-score per language": fig_lang_NER})
    wandb.log({'Global metrics':wandb.Table(data=[list(metrics_dct.values())], columns=list(metrics_dct.keys()))})

    # Move best checkpoint to Weights and Biases root directory to be saved
    log.info(f"Move best checkpoint to Weights and Biases root directory to be saved:")
    os.replace(f"{train_dct['filepath']}/best-checkpoint.bin", f"{wandb.run.dir}/best-checkpoint.bin")
    # End W&B session
    wandb.finish()
   

if __name__=="__main__":
    fire.Fire(main)
