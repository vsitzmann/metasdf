"""
Reconstruct shapes using a trained model.
"""

import sys
sys.path.append('..')
sys.path.append('../..')

import torch
import curriculums
from reconstruction import reconstruct
import levelset_data
import json
import argparse

device = torch.device('cuda')

################################################
context_mode = 'dense'
data_source = '/media/data3/sitzmann/ShapeNetProcessedData/data'

################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--checkpoint', default='latest.pth')
    args = parser.parse_args()
    
    curriculum = getattr(curriculums, args.exp_name)
        
    
    checkpoint = torch.load(curriculum['output_dir'] + '/' + args.checkpoint, map_location=device)
    model = (checkpoint['model'].module)
    reconstruction_output_dir = curriculum['reconstruction_output_dir']

    with open(curriculum['test_split'], "r") as f:
        split = json.load(f)
    npz_filenames = levelset_data.get_instance_filenames(data_source, split)

    reconstruct(model, npz_filenames, reconstruction_output_dir, context_mode=context_mode, data_source=data_source)