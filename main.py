# Requierments
import logging as log
import json
import os
import sys
import torch
import wandb
import fire

# Dependencies
from src.setup import setup_data
from src.dataset import IC_NER_Dataset
from src.model import IC_NER_Model
from src.loss import IC_NER_Loss
from src.fitter import IC_NER_Fitter
from src.metrics import AccIC, F1IC, AccNER, F1NER
from src.callbacks import wandb_checkpoint

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
        setup_config:str="input/setup_config.json",
        model_config:str="input/model_config.json",
        training_config:str="input/training_config.json",
        wandb_config:str="input/wandb_config.json",
        ):

    #
    # Part I: Data Gathering
    #

    # Model dict
    with open(model_config, 'r') as f:
        model_dct = json.load(f)
    # Training dict
    with open(training_config, 'r') as f:
        train_dct = json.load(f)
    # Wandb dict
    with open(wandb_config, 'r') as f:
        wandb_dct = json.load(f)
    # Get data
    log.info(f"Prepare DataLoaders:")
    texts, intents, nlp, intent2idx, ner2idx, MAX_LEN, num_labels = setup_data(setup_config)
    train_dts = IC_NER_Dataset(texts['train'], intents['train'], model_dct['model_name'], MAX_LEN, nlp, intent2idx, ner2idx)
    train_dtl = torch.utils.data.DataLoader(train_dts,
                                            batch_size=train_dct['batch_size'],
                                            num_workers=train_dct['num_workers'],
                                            shuffle=True,
                                            )
    val_dts = IC_NER_Dataset(texts['test'], intents['test'], model_dct['model_name'], MAX_LEN, nlp, intent2idx, ner2idx)
    val_dtl = torch.utils.data.DataLoader(val_dts,
                                          batch_size=2*train_dct['batch_size'],
                                          num_workers=train_dct['num_workers'],
                                          shuffle=False,
                                          )

    #
    # Part II: Model Training
    #

    # Define model
    model = IC_NER_Model(model_dct['model_name'], num_labels, model_dct['dim'], model_dct['dropout'], model_dct['device'])
    # Get loss, optimisers and schedulers
    criterion = IC_NER_Loss(label_smoothing=train_dct['label_smoothing'], n_classes=num_labels, device=model_dct['device'])
    optimizer = torch.optim.AdamW([{'params':model.parameters()}, 
                                  {'params':criterion.parameters()}],
                                  lr=train_dct['learning_rate'],
                                  weight_decay=train_dct['weight_decay'],
                                  )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode=train_dct['scheduler_mode'],
                                                           factor=train_dct['scheduler_factor'],
                                                           patience=train_dct['scheduler_patience'],
                                                           threshold=1e-6,
                                                          )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5*1e-6, last_epoch=- 1, verbose=False)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    # Metrics
    acc_ic = AccIC()
    f1_ic = F1IC()
    acc_ner = AccNER()
    f1_ner = F1NER()
    # Fitter
    if not os.path.isdir(os.path.join(os.getcwd(),train_dct['filepath'])): os.makedirs(os.path.join(os.getcwd(),train_dct['filepath']))
    fitter = IC_NER_Fitter(model,
                           model_dct['device'],
                           criterion,
                           optimizer,
                           scheduler,
                           folder=os.path.join(os.getcwd(),train_dct['filepath']),
                           validation_scheduler = True, # Apply LR scheduler every epoch
                           use_amp = bool(train_dct['use_amp']),
                           )
    # Weights and Biases login
    wandb.login(key=wandb_dct['WB_KEY'])
    wandb.init(project=wandb_dct['WB_PROJECT'], entity=wandb_dct['WB_ENTITY'], config=train_dct)
    wandb.watch(model)
    # Fitter
    log.info(f"Start fitter training:")
    _ = fitter.fit(train_loader = train_dtl,
                   val_loader = val_dtl,
                   n_epochs = train_dct['epochs'],
                   metrics = [(acc_ic, 'acc_ic'), (f1_ic, 'f1_ic'), (acc_ner, 'acc_ner'), (f1_ner, 'f1_ner')],
                   early_stopping = train_dct['early_stopping'],
                   early_stopping_mode = train_dct['scheduler_mode'],
                   verbose_steps = train_dct['verbose_steps'],
                   callbacks = [wandb_checkpoint],
                   )
    # Move best checkpoint to Weights and Biases root directory to be saved
    log.info(f"Move best checkpoint to Weights and Biases root directory to be saved:")
    os.replace(os.path.join(os.getcwd(),train_dct['filepath'],'best-checkpoint.bin'), f"{wandb.run.dir}/best-checkpoint.bin")
    # Finish W&B session
    log.info(f"Finish W&B session:")
    wandb.finish()

if __name__=="__main__":
    fire.Fire(main)
