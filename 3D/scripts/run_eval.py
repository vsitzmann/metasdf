"""
Compute Chamfer distances between reconstructed shapes and ground truth meshes.
Based on code from DeepSDF (Park et al.)
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

import argparse
import json
import numpy as np
import os
import trimesh

def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

import glob

def evaluate(reconstruction_dir, data_dir):
    chamfer_results = []

    reconstruction_paths = glob.glob(f'{reconstruction_dir}/**/**/*.ply', recursive=True)
        
    print(reconstruction_paths[0].split('/'))
    
    
    for reconstructed_mesh_filename in reconstruction_paths:
        print(reconstructed_mesh_filename)
        tokens = reconstructed_mesh_filename.split('/')        
        dataset, class_name, instance_name = tokens[-3], tokens[-2], tokens[-1].split('.')[0]
        

        ground_truth_samples_filename = os.path.join(
            data_dir,
            "SurfaceSamples",
            dataset,
            class_name,
            instance_name + ".ply",
        )

        normalization_params_filename = os.path.join(
            data_dir,
            "NormalizationParameters",
            dataset,
            class_name,
            instance_name + ".npz",
        )

        ground_truth_points = trimesh.load(ground_truth_samples_filename)
        reconstruction = trimesh.load(reconstructed_mesh_filename)

        normalization_params = np.load(normalization_params_filename)

        chamfer_dist = compute_trimesh_chamfer(
            ground_truth_points,
            reconstruction,
            normalization_params["offset"],
            normalization_params["scale"],
        )
        
        print(chamfer_dist)

        chamfer_results.append(chamfer_dist)
                
    print(f'\n\n\n\nMean: {np.mean(chamfer_results)} \t Median: {np.median(chamfer_results)} \t Stddev: {np.std(chamfer_results)}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('reconstruction_dir')
    parser.add_argument('--data_dir', default='/media/data3/sitzmann/ShapeNetProcessedData/data')
    args = parser.parse_args()
    
    evaluate(args.reconstruction_dir, args.data_dir)