# Requierments
import os
import json
import pandas as pd
import numpy as np
import torch
import argparse
import fire
from pathlib import Path
# Dependencies
from .src.setup import setup_data
from .src.dataset import MT_IC_HNER_Dataset
from .src.model import MT_IC_HNER_Model
from .src.loss import MT_IC_HNER_Loss
from .src.metrics import metrics
from .src.fitter import MT_IC_HNER_Fitter


# Method to gather all arguments
def parse_args():

    parser = argparse.ArgumentParser(description='Template')

    # === PATHS === #
    parser.add_argument('--setup_config', type=str, default="input/setup_config.json",
                                            help='Path to configuration file for data extraction and arrangement of both intents and entities.')
    parser.add_argument('--model_config', type=str, default="input/model_config.json",
                                            help='Path to configuration file for model hyperparameters.')
    parser.add_argument('--training_config', type=str, default="input/training_config.json",
                                            help='Path to configuration file for training hyperparameters.')
    parser.add_argument('--metrics_config', type=str, default="input/metrics_config.json",
                                            help='Path to configuration file for intent classifier multilabel metrics hyperparameters.')



def main():
    # Fetch args
    args = parse_args()
    with open(args['--model_config'], 'r') as f:
        model_config = json.load(f)
    with open(args['--training_config'], 'r') as f:
        training_config = json.load(f)
    with open(args['--metrics_config'], 'r') as f:
        metrics_config = json.load(f)

    # Run setup
    texts, multilabel_intents, tags, nlp, tag2idxs, idxs2tag, original_idxs2tag, MAX_LEN, num_labels = setup_data(setup_config_path=args['--setup_config'])
    # Build torch dataloaders
    dts_train = MT_IC_HNER_Dataset(texts['train'], multilabel_intents['train'], model_config['model_name'], MAX_LEN, nlp, tag2idxs)
    dtl_train = torch.utils.data.DataLoader(dts_train, batch_size = training_config['batch_size'], num_workers = 2, shuffle=True)
    dts_val = MT_IC_HNER_Dataset(texts['test'], multilabel_intents['test'], model_config['model_name'], MAX_LEN, nlp, tag2idxs)
    dtl_val = torch.utils.data.DataLoader(dts_val, batch_size = 2*training_config['batch_size'], num_workers = 2, shuffle=False)
    # Define model
    model = MT_IC_HNER_Model(model_name=model_config['model_name'],
                             num_labels=num_labels,
                             proj_dim=model_config['proj_dim'],
                             num_heads=model_config['num_heads'],
                             hidden_dropout_prob=model_config['hidden_dropout_prob'],
                             layer_norm_eps=model_config['layer_norm_eps'],
                             dropout=model_config['dropout'],
                             device=model_config['device'],
                             )
    # Get loss, optimisers and schedulers
    criterion = MT_IC_HNER_Loss(tags, reduction='mean', label_smoothing=training_config['label_smoothing'])
    optimizer = torch.optim.AdamW([{'params':model.parameters()}, 
                                  {'params':criterion.parameters()}], lr=training_config['learning_rate'], weight_decay = training_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=training_config['early_stopping']//2)
    # Get metrics parameters (overwrite IC classes just in case)
    metrics_config['num_classes'] = num_labels['IC']

    # Fitter
    fitter = MT_IC_HNER_Fitter(metrics_config,
                               idxs2tag,
                               original_idxs2tag,
                               model,
                               model_config['device'],
                               criterion,
                               optimizer,
                               scheduler,
                               folder=os.path.join(training_config['filepath'], 'models'),
                               validation_scheduler = True, # Apply LR scheduler every epoch
                               )

    history = fitter.fit(train_loader = dtl_train,
                     val_loader = dtl_val,
                     n_epochs = training_config['epochs'],
                     metrics = None,#metrics,
                     early_stopping = training_config['early_stopping'],
                     verbose_steps = training_config['verbose_steps'],
                     )
    
    # Save metrics results
    history.to_csv(os.path.join(training_config['filepath'], 'training_history.csv'))

if __name__ == "__main__":
    fire.Fire(main)