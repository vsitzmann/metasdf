#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import sys
import levelset_data

torch.backends.cudnn.benchmark = True

from meta_modules import *
from modules import *

max_batch = 2000000  # Make as big as your memory allows
N=256

def reconstruct(decoder, npz_filenames, reconstruction_dir, context_mode, data_source, skip=True, test_time_optim_steps=None):
    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    for ii, npz in enumerate(npz_filenames):
        filename = os.path.splitext(npz)[0]
        instance_name = '/'.join(filename.split('/')[1:])
        sdf_filename = os.path.join(data_source, 'SdfSamples', filename) + '.npz'
        levelset_filename = os.path.join(data_source, 'SurfaceSamples', filename) + '.ply'
        partial_filename = os.path.join('/home/ericryanchan/depth_maps', instance_name, 'world_coords.ply')
        normalization_params_filename = os.path.join(data_source, 'NormalizationParameters', filename) + '.npz'
        if ("npz" not in npz or
            not os.path.exists(sdf_filename)):
            print("SDF Samples not found")
            continue
        if context_mode == 'levelset' and (not os.path.exists(levelset_filename) or not os.path.exists(normalization_params_filename)):
            print("Levelset not found: ", levelset_filename)
            continue
        if context_mode == 'partial' and (not os.path.exists(partial_filename) or not os.path.exists(normalization_params_filename)):
            print("Partial not found: ", partial_filename)
            continue
            
        mesh_filename = os.path.join(reconstruction_dir, npz[:-4])
        ply_filename = mesh_filename + '.ply'

        try:
            reconstructed_sdf = generate_dense_cube(decoder, sdf_filename, levelset_filename, partial_filename, normalization_params_filename, context_mode, test_time_optim_steps=test_time_optim_steps)
        except OSError as e:
            print(e)
            #print('OS Error?')
            continue

        ############ Output to PLY ##############
        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)


        levelset_data.convert_sdf_samples_to_ply(
            reconstructed_sdf.data,
            voxel_origin,
            voxel_size,
            ply_filename,
            offset=None,
            scale=None,
            level=0.0
        )

def generate_dense_cube(decoder, sdf_filename, levelset_filename, partial_filename, normalization_params_filename, context_mode, test_time_optim_steps=None):
    data_sdf = levelset_data.read_sdf_samples_into_ram(sdf_filename, levelset_filename, partial_filename, normalization_params_filename, context_mode=context_mode)

    ####### Set up data #########

    sampled_data = levelset_data.unpack_sdf_samples_from_ram(data_sdf, subsampleSDF = 100000, subsampleLevelset = 10000)

    if context_mode == 'partial':
        meta_data = levelset_data.meta_split(sampled_data['sdf'].unsqueeze(0),
                                             sampled_data['partial'].unsqueeze(0),
                                             context_mode=context_mode)
    else:
        meta_data = levelset_data.meta_split(sampled_data['sdf'].unsqueeze(0),
                                             sampled_data['levelset'].unsqueeze(0),
                                                 context_mode=context_mode)

    ####### Use the given SDF samples as context to adapt the meta-network ##########
    context_x = meta_data['context'][0].cuda()
    context_y = meta_data['context'][1].cuda()

    with torch.no_grad():
        start_time = time.time()
        params = decoder.generate_params(context_x, context_y, intermediate=False, num_meta_steps=test_time_optim_steps)

        print(f"Adaptation in {time.time() - start_time} seconds")

        ###### Reconstruct mesh by sampling densely from a 256^3 cube ###########
        reconstruction_points = levelset_data.create_samples(N=N).cuda()
        reconstructed_sdf = torch.zeros((reconstruction_points.shape[0], 1)).cpu()

        decoder.eval()
        head = 0
        while head < reconstruction_points.shape[0]:
            query_x = reconstruction_points[head:min(head + max_batch, reconstruction_points.shape[0]), 0:3].unsqueeze(0)

            # When reconstructing intermediate steps, loop through parameters of each iteration.
            predictions = decoder.forward_with_params(query_x, params).detach()
            predictions = predictions.squeeze(0)

            if predictions.shape[-1] == 2:
                sdf = torch.sign(predictions[:, 0:1]) * torch.abs(predictions[:, 1:2]) # If using composite loss
#                 sdf = predictions[:, 1:2]
            else:
                sdf = predictions

            reconstructed_sdf[head:min(head + max_batch, reconstruction_points.shape[0]), 0] = sdf.squeeze(1).detach().cpu()
            head += max_batch
    reconstructed_sdf = reconstructed_sdf.reshape(N, N, N)
    return reconstructed_sdf