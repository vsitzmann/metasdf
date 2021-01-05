import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import glob
import os
from torch.utils.data import DataLoader
import statistics
from tqdm.autonotebook import tqdm
import shutil
import imageio
from torchvision.utils import make_grid
import modules


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_meta_summaries(model_output, meta_batch, writer, total_steps, prefix):
    gt_sds = dataio.lin2img(meta_batch['test'][1]).squeeze().cpu()
    pred_sds = dataio.lin2img(model_output).squeeze().detach().cpu()

    valid_levelset_points = (meta_batch['train'][1] == 0.).repeat(1,1,2)
    if valid_levelset_points.any():
        level_set = meta_batch['train'][0]
        batch_size = level_set.shape[0]
        fig, axes = plt.subplots(int(batch_size / 8), 8)
        levelset_points = level_set.detach().cpu().numpy()
        for i in range(batch_size):
            num_level_set_points = (meta_batch['train'][1][i] == 0.).shape[0]
            digit = levelset_points[i, :num_level_set_points, :]
            im = axes[i // 8, i % 8].scatter(digit[:, 1], -digit[:, 0])
            axes[i // 8, i % 8].axis('off')
        plt.tight_layout()

        writer.add_figure(prefix + 'levelset', fig, global_step=total_steps)

    output_vs_gt = torch.cat((gt_sds, pred_sds), dim=-1)[:,None,...]
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    writer.add_scalar(prefix + 'gt_min', gt_sds.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(prefix + 'gt_max', gt_sds.max().detach().cpu().numpy(), total_steps)

    writer.add_scalar(prefix + 'pred_min', pred_sds.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(prefix + 'pred_max', pred_sds.max().detach().cpu().numpy(), total_steps)

    writer.add_scalar(prefix + 'dense_coords_min', meta_batch['test'][0].min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(prefix + 'dense_coords_max', meta_batch['test'][0].min().detach().cpu().numpy(), total_steps)
    
    
def write_summaries(model_output, model_input, gt, writer, total_steps, prefix):
    gt_sds = dataio.lin2img(gt['sds']).squeeze().cpu()
    pred_sds = dataio.lin2img(model_output).squeeze().detach().cpu()

    """
    # plot level sets
    batch_size = model_input['level_set'].shape[0]
    fig, axes = plt.subplots(-(-batch_size // 8), 8)
    levelset_points = model_input['level_set'].detach().cpu().numpy()
    
    if batch_size > 1:
        for i in range(batch_size):
            num_level_set_points = (gt['ls_sds'][i] == 0.).shape[0]
            digit = levelset_points[i, :num_level_set_points, :]

            im = axes[i//8, i%8].scatter(digit[:,1], -digit[:,0])
            axes[i//8, i%8].axis('off')
        plt.tight_layout()
    else:
        num_level_set_points = (gt['ls_sds'][0] == 0.).shape[0]
        digit = levelset_points[0, :num_level_set_points, :]
        im = axes[0].scatter(digit[:,1], -digit[:,0])
        axes[0].axis('off')

    writer.add_figure(prefix + 'levelset', fig, global_step=total_steps)
    """
    
    """
    output_vs_gt = torch.cat((gt_sds, pred_sds), dim=-1)[:,None,...]
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    """

    writer.add_scalar(prefix + 'gt_min', gt_sds.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(prefix + 'gt_max', gt_sds.max().detach().cpu().numpy(), total_steps)

    writer.add_scalar(prefix + 'pred_min', pred_sds.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(prefix + 'pred_max', pred_sds.max().detach().cpu().numpy(), total_steps)

    writer.add_scalar(prefix + 'dense_coords_min', gt['sds'].min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(prefix + 'dense_coords_max', gt['sds'].min().detach().cpu().numpy(), total_steps)
    writer.flush()


def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)

def plot_mnist_digit(digit):
    num_pixels = digit.shape[1]
    sidelen = int(np.sqrt(num_pixels))
    plt.imshow(digit.detach().cpu().numpy().reshape(sidelen,sidelen))
    plt.show()


def plot_sds(gt_sd, pred_sd):
    # Images are square, but flattened - compute the sidelength.
    num_pixels = gt_sd.shape[1]
    sidelen = int(np.sqrt(num_pixels))

    pred_sd = pred_sd[0, ...].detach().cpu().numpy()
    pred_sd = pred_sd.reshape(sidelen, sidelen)

    gt_sd = gt_sd[0, ...].detach().cpu().numpy()
    gt_sd = gt_sd.reshape(sidelen, sidelen)

    fig, (axa, axb) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    axa.cla(), axb.cla()

    # PLOT A: Ground truth SDs
    axa.imshow(gt_sd)
    axa.contour(gt_sd, levels=[0], colors='k', linestyles='-')
    axa.set_title('Ground truth signed Distance')

    # PLOT A: Predicted SDs
    axb.imshow(pred_sd)
    axb.contour(pred_sd, levels=[0], colors='k', linestyles='-')
    axb.set_title('Predicted Signed Distance')

    plt.show()


def plot_sds_with_gradients(gt_contour, gt_normals, mgrid_sds, mgrid_grads):
    '''Plot signed distances with gradients.

    gt_contour: np array of shape (-1, 2) with x,y coordinates of gt contour.
    mgrid_sds: signed distances evaluated on square meshgrid.
    mgrid_grads: grads evaluated on square meshgrid.
    '''
    # Images are square, but flattened - compute the sidelength.
    num_pixels = mgrid_grads.shape[1]
    sidelen = int(np.sqrt(num_pixels))
    
    mgrid = dataio.get_mgrid(sidelen).detach().cpu().numpy()
    x = np.linspace(-1, 1, sidelen)
    y = np.linspace(-1, 1, sidelen)
    
    
    mgrid_grads = mgrid_grads[0,...].detach().cpu().numpy()
    
    mgrid_grads_mag = np.linalg.norm(mgrid_grads, axis=-1)
    
    gt_contour = gt_contour[0,...].detach().cpu().numpy()  
    gt_normals = gt_normals[0,...].detach().cpu().numpy()
    
    mgrid_sds = mgrid_sds[0,...].detach().cpu().numpy()  
    mgrid_sds = mgrid_sds.reshape(sidelen, sidelen)

    fig, (axa, axb, axc) = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
    axa.cla(), axb.cla(), axc.cla()

    # PLOT A: Ground truth mesh
    axa.set_xlim([mgrid.min(), mgrid.max()]), axa.set_ylim([mgrid.min(), mgrid.max()])
    axa.plot(gt_contour[...,1], gt_contour[...,0]*-1.)
    q = axa.quiver(gt_contour[...,1], gt_contour[...,0]*-1., 
                   gt_normals[..., 1], gt_normals[..., 0]*-1., scale=25.)
    axa.set_title('Ground truth level set')
    
    # PLOT A: Predicted SDs
    axb.imshow(mgrid_sds)
    axb.contour(mgrid_sds, levels=[0], colors='k', linestyles='-')
    axb.set_title('Predicted Signed Distance')
    
    # PLOT B: GRADIENT DIRECTIONS
    grad_subsample = 1
    quiver_coords = mgrid.reshape(sidelen, sidelen, 2)[::grad_subsample,::grad_subsample,:].reshape(-1, 2)
    quiver_mgrid_grads = mgrid_grads.reshape(sidelen, sidelen, 2)[::grad_subsample,::grad_subsample,:].reshape(-1, 2)
    
    axc.set_xlim([mgrid.min(), mgrid.max()]), axc.set_ylim([mgrid.min(), mgrid.max()])
    axc.set_xlabel('x'), axc.set_ylabel('y')
    axc.set_xticks([mgrid.min(), 0., mgrid.max()]), axc.set_yticks([mgrid.min(), 0., mgrid.max()])

    q = axc.quiver(quiver_coords[...,1], quiver_coords[...,0] * -1, 
               quiver_mgrid_grads[..., 1], quiver_mgrid_grads[..., 0] * -1)
    
    axc.set_title('Orientations of Gradients')

    plt.show()


def plot_loss_curve(model_dir):
    # Plot loss curve
    latest_history_path = max(glob.glob(os.path.join(model_dir, 'train_losses*.txt')), key=os.path.getctime)
    train_losses = np.loadtxt(latest_history_path).tolist()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(train_losses)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    
    # Find median of last 1000 losses
    final_loss = statistics.median(train_losses[-1000:])
    ax.set_title(f'Train Losses, final loss: {final_loss}')
    
    
def plot_unique_examples(model, dataset, unique_examples=4):
    # Grabs four random examples from the dataset and plots predictions against ground truths
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=unique_examples, pin_memory=False, sampler=None, num_workers=0)
    
    ground_truth_plots = []
    prediction_plots = []    
    
    model_input, gts = next(iter(dataloader))
    model_input = {key: value.cuda() for key, value in model_input.items()}
    pred_sds, _ = model.forward(**model_input)
    
    prediction_plots = pred_sds.detach().cpu().numpy()
    ground_truth_plots = gts['sds'].cpu().numpy()
    
    fig, axes = plt.subplots(2, len(ground_truth_plots), figsize=(10, 6))
    for index, (gt_sd, pred_sd) in enumerate(zip(ground_truth_plots, prediction_plots)):
        axa = axes[0][index]
        axb = axes[1][index]
        
        num_pixels = gt_sd.shape[0]
        sidelen = int(np.sqrt(num_pixels))

        pred_sd = pred_sd.reshape(sidelen, sidelen)
        gt_sd = gt_sd.reshape(sidelen, sidelen)


        # PLOT A: Ground truth SDs
        axa.imshow(gt_sd)
        axa.contour(gt_sd, levels=[0], colors='k', linestyles='-')
        axa.set_title('Ground truth')

        # PLOT A: Predicted SDs
        axb.imshow(pred_sd)
        axb.contour(pred_sd, levels=[0], colors='k', linestyles='-')
        axb.set_title('Prediction')


def latest_state_dict(model_dir):
    latest_path = max(glob.glob(os.path.join(model_dir, 'checkpoint*.pth')), key=os.path.getctime)
    return torch.load(latest_path)

                      
def evaluate_model(model, dataloader):
    losses = []
    saved_out = 0
    for idx, (model_input, gt) in enumerate(dataloader):
        with torch.no_grad():
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            pred_sd, z = model.legacy_forward(**model_input)
            pred_sd = pred_sd.detach().cpu()

            loss = modules.sdf_loss(pred_sd, gt['sds'].cpu())
            losses.append(loss.item())
            
    return np.mean(losses)
