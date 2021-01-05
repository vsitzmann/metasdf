import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import modules
import time
import numpy as np
import os
import glob
import re
import shutil


def train_with_signed_distance(model,
                               train_dataloader,
                               val_dataloader,
                               epochs,
                               lr,
                               steps_til_summary,
                               epochs_til_checkpoint,
                               model_dir,
                               supervision='dense'):
    
    assert (supervision in ['levelset', 'dense'])
    
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                
                if supervision=='levelset': # Use level_set only
                    model_input_level_set = model_input.copy()
                    model_input_level_set['coords'] = model_input_level_set['level_set']
                    model_input_level_set = {key: value.cuda() for key, value in model_input_level_set.items()}
                    pred_sd, z = model.legacy_forward(**model_input_level_set)
                    loss = modules.sdf_loss(pred_sd, gt['ls_sds']) + torch.mean(z ** 2)
                elif supervision=='dense': # Use standard coords
                    pred_sd, z = model.legacy_forward(**model_input)
                    loss = modules.sdf_loss(pred_sd, gt['sds']) + torch.mean(z ** 2)

                train_losses.append(loss.item())
                writer.add_scalar("train_loss", loss, total_steps)

                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    corrected_loss = utils.evaluate_model(model, train_dataloader)
                    writer.add_scalar('corrected_loss', corrected_loss, total_steps)
                    pred_sd, z = model.legacy_forward(**model_input)
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, corrected_loss, time.time() - start_time))
                    utils.write_summaries(pred_sd, model_input, gt, writer, total_steps, 'train_')
                    
                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for meta_batch in val_dataloader:
                                pred_sd = model(meta_batch)
                                val_loss = modules.sdf_loss(pred_sd, meta_batch['test'][1].cuda())
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                            utils.write_meta_summaries(pred_sd, meta_batch, writer, total_steps, 'val_')
                        model.train()

                total_steps += 1

            if not epoch % epochs_til_checkpoint and epoch:
                
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                           np.array(train_losses))

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

               

                

def train_with_signed_distance_meta(model,
                                    train_dataloader,
                                    val_dataloader,
                                    epochs,
                                    lr,
                                    steps_til_summary,
                                    epochs_til_checkpoint,
                                    model_dir):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f'\n\nTraining model with {num_parameters} parameters\n\n')

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            for step, meta_batch in enumerate(train_dataloader):
                start_time = time.time()
                pred_sd, _ = model(meta_batch)
                loss = modules.sdf_loss(pred_sd, meta_batch['test'][1].cuda())

                train_losses.append(loss)
                writer.add_scalar("train_loss", loss, total_steps)

                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.update(1)

                tqdm.write(
                    "Epoch %d, Total loss %0.6f, iteration time %0.6f" %
                    (epoch, loss, time.time() - start_time))

                if not total_steps % steps_til_summary:
                    utils.write_meta_summaries(pred_sd, meta_batch, writer, total_steps, 'train_')

                    print("Running validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for val_idx, meta_batch in enumerate(val_dataloader):
                            pred_sd, _ = model(meta_batch)
                            val_loss = modules.sdf_loss(pred_sd, meta_batch['test'][1].cuda())
                            val_losses.append(val_loss.cpu().numpy())

                            if not val_idx:
                                utils.write_meta_summaries(pred_sd, meta_batch, writer, total_steps, 'val_')

                        writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        tqdm.write("Validation loss %0.6e" % loss)
                    model.train()

                total_steps += 1

            if not epoch % epochs_til_checkpoint:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'epoch_%03d.pth'%epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%03d.txt'%epoch),
                           np.array(train_losses))

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))



