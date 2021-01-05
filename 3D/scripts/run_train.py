"""
Start a training run with an experiment.

"""

import torch
import torch.utils.data as data_utils
import signal
import os
import math
import json
import sys
import numpy as np

sys.path.append('..')
sys.path.append('../..')
from levelset_data import LevelsetDataset
import levelset_data
import argparse
import curriculums

from training import train

device = torch.device('cuda')

#########################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--load', action='store_true')
    
    args = parser.parse_args()
    
    curriculum = getattr(curriculums, args.exp_name)        
    
    if not os.path.isdir(curriculum['output_dir']):
        os.makedirs(curriculum['output_dir'])
    
    model = torch.nn.DataParallel(curriculum['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=curriculum['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=0.5)
    
    with open(curriculum['train_split'], "r") as f:
        train_split = json.load(f)
    with open(curriculum['val_split'], "r") as f:
        val_split = json.load(f)
        
    dataset = LevelsetDataset(
        curriculum['data_source'], train_split, context_mode=curriculum['context_mode'], subsampleSDF=curriculum['SDFSamplesPerScene'], subsampleLevelset=curriculum['LevelsetSamplesPerScene'], load_ram=False
    )
    val_dataset = LevelsetDataset(
        curriculum['data_source'], val_split, context_mode=curriculum['context_mode'], subsampleSDF=curriculum['SDFSamplesPerScene'], subsampleLevelset=curriculum['LevelsetSamplesPerScene'], load_ram=False
    )
        
    dataloader = data_utils.DataLoader(
        dataset,
        batch_size=curriculum['ScenesPerBatch'],
        shuffle=True,
        num_workers=16,
        drop_last=False,
    )
    val_dataloader = data_utils.DataLoader(
        val_dataset,
        batch_size=curriculum['ScenesPerBatch'],
        shuffle=True,
        num_workers=16,
        drop_last=False,
    )

    start_epoch = 1

    if args.load:
        checkpoint = torch.load(curriculum['output_dir'] + '/latest.pth', map_location=device)
        model = (checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    train(model, optimizer, scheduler, dataloader, start_epoch, curriculum['num_epochs'], curriculum['training_mode'], curriculum['context_mode'], output_dir=curriculum['output_dir'], val_dataloader=val_dataloader)