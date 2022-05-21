# Requirements
from typing import Dict
import wandb

# Wandb callback
def wandb_checkpoint(history:Dict):
    train_checkpoint = history.copy()
    del train_checkpoint['epoch']
    wandb.log(train_checkpoint)
