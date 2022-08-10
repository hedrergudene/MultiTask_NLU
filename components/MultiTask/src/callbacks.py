# Requirements
import wandb
from typing import Dict

# Weights and Biases callback
def wandb_checkpoint(history:Dict, step:int):
    train_checkpoint = {k:v for k,v in history.items() if k!='epoch'}
    wandb.log(train_checkpoint, step=step)