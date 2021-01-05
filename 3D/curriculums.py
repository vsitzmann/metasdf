"""
Curriculums for 3D experiments!
Add your own experiments or adjust settings here.

planes_relu, etc. are similar to the experiments used in the paper.
planes_pe, etc. use positional encodings (Mildenhall et al.) and smaller networks.
They generally perform comparably to relu models but are less memory intensive.
"""

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import time
import torch
import os

from meta_modules import *
from modules import *

def inner_maml_l1_loss(predictions, gt, sigma=None):
    return torch.abs(predictions - gt).sum(0).mean()

def inner_maml_multitask_loss(pred_input, gt_sdf, sigma):
    gt_sign = (gt_sdf > 0).float()
    pred_sign = torch.sigmoid(pred_input[:, :, 0:1])
    pred_sdf = pred_input[:, :, 1:2]
    
    bce_loss = torch.nn.BCELoss(reduction='none')(pred_sign, gt_sign).sum(0).mean()
    l1_loss = torch.abs(pred_sdf - gt_sdf).sum(0).mean()
        
    l = bce_loss/(2 * sigma[0]**2) + l1_loss/(2 * sigma[1]**2) + torch.log(sigma.prod())
    return l

def PEMetaSDF():
        hypo_module = PEFC(in_features=3, out_features=2,
                         num_hidden_layers=6, hidden_features=512)
        hypo_module.apply(sal_init)
        hypo_module.net[-1].apply(sal_init_last_layer)

        model = MetaSDF(hypo_module, inner_maml_multitask_loss, num_meta_steps=3, init_lr=5e-3,
                        lr_type='simple_per_parameter', first_order=False)

        return model
    
def ReLUMetaSDF():
        hypo_module = ReLUFC(in_features=3, out_features=2,
                         num_hidden_layers=8, hidden_features=512)
        hypo_module.apply(sal_init)
        hypo_module.net[-1].apply(sal_init_last_layer)

        model = MetaSDF(hypo_module, inner_maml_multitask_loss, num_meta_steps=5, init_lr=5e-3,
                        lr_type='per_parameter', first_order=False)

        return model

planes_relu = {
    'model': ReLUMetaSDF(),
    'train_split': '../splits/planes_train.json', # Train split file, following DeepSDF's format
    'val_split': '../splits/planes_test.json',    # Val split if you have one
    'test_split': '../splits/planes_test.json',   # Test split, used only for reconstruction
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data', # Preprocessed data folder, following DeepSDF
    'num_epochs': 3000,
    'training_mode': 'multitask', # Either 'multitask' for composite loss or 'l1' for simple loss
    'context_mode': 'dense', # Either 'dense' or 'levelset'
    'SDFSamplesPerScene': 24000, # Number of SDF samples drawn per scene, per iteration
    'LevelsetSamplesPerScene': 0, # Number of Levelset samples drawn per scene, per iteration. Only needed for 'levelset' training.
    'ScenesPerBatch': 32, # Meta-batch size
    'output_dir': 'model_parameters/planes_relu', # Directory to save parameters and tensorboard files
    'reconstruction_output_dir': 'reconstructions/planes_relu', # Directory to save reconstructions, only used during reconstruction
}

tables_relu = {
    'model': ReLUMetaSDF(),
    'train_split': '../splits/tables_train.json',
    'val_split': '../splits/tables_test.json',
    'test_split': '../splits/tables_test.json',
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data',
    'num_epochs': 3000,
    'training_mode': 'multitask',
    'context_mode': 'dense',
    'SDFSamplesPerScene': 24000,
    'LevelsetSamplesPerScene': 0,
    'ScenesPerBatch': 32,
    'output_dir': 'model_parameters/tables_relu',
    'reconstruction_output_dir': 'reconstructions/tables_relu',
    'lr': 5e-4,
}

benches_relu = {
    'model': ReLUMetaSDF(),
    'train_split': '../splits/benches_train.json',
    'val_split': '../splits/benches_test.json',
    'test_split': '../splits/benches_test.json',
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data',
    'num_epochs': 3000,
    'training_mode': 'multitask',
    'context_mode': 'dense',
    'SDFSamplesPerScene': 24000,
    'LevelsetSamplesPerScene': 0,
    'ScenesPerBatch': 32,
    'output_dir': 'model_parameters/benches_relu',
    'reconstruction_output_dir': 'reconstructions/benches_relu',
    'lr': 5e-4,
}

planes_pe = {
    'model': PEMetaSDF(),
    'train_split': '../splits/planes_train.json',
    'val_split': '../splits/planes_test.json',
    'test_split': '../splits/planes_test.json',
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data',
    'num_epochs': 3000,
    'training_mode': 'multitask',
    'context_mode': 'dense',
    'SDFSamplesPerScene': 8000,
    'LevelsetSamplesPerScene': 0,
    'ScenesPerBatch': 28,
    'output_dir': 'model_parameters/planes_pe',
    'reconstruction_output_dir': 'reconstructions/planes_pe',
    'lr': 5e-4,
}

tables_pe = {
    'model': PEMetaSDF(),
    'train_split': '../splits/tables_train.json',
    'val_split': '../splits/tables_test.json',
    'test_split': '../splits/tables_test.json',
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data',
    'num_epochs': 3000,
    'training_mode': 'multitask',
    'context_mode': 'dense',
    'SDFSamplesPerScene': 8000,
    'LevelsetSamplesPerScene': 0,
    'ScenesPerBatch': 28,
    'output_dir': 'model_parameters/tables_pe',
    'reconstruction_output_dir': 'reconstructions/tables_pe',
    'lr': 5e-4,
}

benches_pe = {
    'model': PEMetaSDF(),
    'train_split': '../splits/benches_train.json',
    'val_split': '../splits/benches_test.json',
    'test_split': '../splits/benches_test.json',
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data',
    'num_epochs': 3000,
    'training_mode': 'multitask',
    'context_mode': 'dense',
    'SDFSamplesPerScene': 8000,
    'LevelsetSamplesPerScene': 0,
    'ScenesPerBatch': 28,
    'output_dir': 'model_parameters/benches_pe',
    'reconstruction_output_dir': 'reconstructions/benches_pe',
    'lr': 5e-4,
}

benches_debug = {
    'model': PEMetaSDF(),
    'train_split': '../splits/benches_train.json',
    'val_split': '../splits/benches_test.json',
    'test_split': '../splits/benches_test.json',
    'data_source': '/media/data3/sitzmann/ShapeNetProcessedData/data',
    'num_epochs': 3000,
    'training_mode': 'multitask',
    'context_mode': 'dense',
    'SDFSamplesPerScene': 8000,
    'LevelsetSamplesPerScene': 0,
    'ScenesPerBatch': 32,
    'output_dir': 'model_parameters/benches_debug',
    'reconstruction_output_dir': 'reconstructions/benches_debug',
    'lr': 1e-3,
}