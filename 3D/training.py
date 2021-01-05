#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import os
import math
import json
import sys
import numpy as np
from tqdm.autonotebook import tqdm

sys.path.append('..')
from levelset_data import LevelsetDataset
import levelset_data
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.benchmark = True


def train_epoch(model, dataloader, training_mode, context_mode, optimizer):
    model.train()
    epoch_train_misclassification_percentage = 0
    epoch_train_loss = 0

    for data_dict, indices in dataloader:
        sdf_tensor = data_dict['sdf'].cuda()
        levelset_tensor = data_dict['levelset'].cuda()

        sdf_tensor.requires_grad = False
        levelset_tensor.requires_grad = False

        meta_data = levelset_data.meta_split(sdf_tensor, levelset_tensor, context_mode)
        
        prediction, _ = model(meta_data)
#         context_x, context_y = meta_batch['context']
#         query_x, query_y = meta_batch['query']
        
#         fast_params = model.generate_params(context_x, context_y)
#         prediction = model(query_x, fast_params)
        query_y = meta_data['query'][1]

        if training_mode == 'multitask':
            gt_sign = (query_y > 0).float()
            pred_sign = torch.sigmoid(prediction[:, :, 0:1])
            pred_sdf = prediction[:, :, 1:2]

            bce_loss = torch.nn.BCELoss(reduction='none')(pred_sign, gt_sign).mean()
            l1_loss = torch.abs(torch.where(query_y!=-1., pred_sdf - query_y, torch.zeros_like(pred_sdf))).mean()

            sigma = model.module.sigma_outer

            batch_loss = bce_loss/(2 * sigma[0]**2) + l1_loss/(2 * sigma[1]**2) + torch.log(sigma.prod())

            epoch_train_misclassification_percentage += (torch.sum(torch.sign(prediction[:, :, 0:1]) !=
                torch.sign(query_y)).float()/ (query_y.shape[0]*query_y.shape[1])).detach().cpu().item()

        elif training_mode == 'l1':
            pred_sdf = prediction

            l1_loss = torch.abs(pred_sdf - test_gt).mean()
            batch_loss = l1_loss

            epoch_train_misclassification_percentage += (torch.sum(torch.sign(prediction[:, :, 0:1]) !=
                    torch.sign(test_gt)).float()/ (test_gt.shape[0]*test_gt.shape[1])).detach().cpu().item()

        else:
            raise NotImplementedError

        print(batch_loss.item())
        epoch_train_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()

    epoch_train_misclassification_percentage/=len(dataloader)
    epoch_train_loss/=len(dataloader)
    
    return epoch_train_loss, epoch_train_misclassification_percentage
    
def val_epoch(model, dataloader, training_mode, context_mode):
    epoch_misclassification_percentage = 0
    epoch_loss = 0

    model.eval()
    for data_dict, indices in dataloader:
        with torch.no_grad():
            sdf_tensor = data_dict['sdf'].cuda()
            levelset_tensor = data_dict['levelset'].cuda()

            meta_data = levelset_data.meta_split(sdf_tensor, levelset_tensor, context_mode)            
            
#             context_x, context_y = meta_batch['context']
#             query_x, query_y = meta_batch['query']
            query_y = meta_data['query'][1]
        
        
            prediction, _ = model(meta_data)
            # fast_params = model.generate_params(context_x, context_y)
            # prediction = model(query_x, fast_params)

            if training_mode == 'multitask':
                gt_sign = (query_y > 0).float()
                pred_sign = torch.sigmoid(prediction[:, :, 0:1])
                pred_sdf = prediction[:, :, 1:2]

                bce_loss = torch.nn.BCELoss(reduction='none')(pred_sign, gt_sign).mean()
                l1_loss = torch.abs(torch.where(query_y!=-1., pred_sdf - query_y, torch.zeros_like(pred_sdf))).mean()

                sigma = model.module.sigma_outer

                batch_loss = bce_loss/(2 * sigma[0]**2) + l1_loss/(2 * sigma[1]**2) + torch.log(sigma.prod())

                epoch_misclassification_percentage += (torch.sum(torch.sign(prediction[:, :, 0:1]) !=
                    torch.sign(query_y)).float()/ (query_y.shape[0]*query_y.shape[1])).detach().cpu().item()

            elif training_mode == 'l1':
                pred_sdf = prediction

                l1_loss = torch.abs(pred_sdf - query_y).mean()
                batch_loss = l1_loss

                epoch_misclassification_percentage += (torch.sum(torch.sign(prediction[:, :, 0:1]) !=
                        torch.sign(query_y)).float()/ (query_y.shape[0]*query_y.shape[1])).detach().cpu().item()

            else:
                raise NotImplementedError

            print(batch_loss.item())
            epoch_loss += batch_loss.item()

    epoch_misclassification_percentage/=len(dataloader)
    epoch_loss/=len(dataloader)
    return epoch_loss, epoch_misclassification_percentage

def train(model, optimizer, scheduler, dataloader, start_epoch, num_epochs, training_mode, context_mode, output_dir='./model_parameters/', save_freq=100, val_dataloader=None):
    writer = SummaryWriter(output_dir)
        
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        
        epoch_train_loss, epoch_train_misclassification_percentage = train_epoch(model, dataloader, training_mode, context_mode, optimizer)
        epoch_val_loss, epoch_val_misclassification_percentage = val_epoch(model, val_dataloader, training_mode, context_mode)
        
        scheduler.step()


        tqdm.write(f"Epoch: {epoch} \t Train Misclassified Percentage: {epoch_train_misclassification_percentage} \t Val Misclassified Percentage: {epoch_val_misclassification_percentage}\t {output_dir}")
        
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Misclassified Percentage/Train', epoch_train_misclassification_percentage, epoch)
        writer.add_scalar('Misclassified Percentage/Val', epoch_val_misclassification_percentage, epoch)
        
        torch.save({'model': model,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch}, os.path.join(output_dir, 'latest.pth'))
        if epoch % save_freq == 0:
            torch.save({'model': model,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch}, os.path.join(output_dir, f'{epoch:04d}.pth'))