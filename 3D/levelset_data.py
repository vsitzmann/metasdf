"""
Dataloaders for 3D experiments. Based on code from DeepSDF (Park et al.)

"""
import time
import plyfile
import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import trimesh
import skimage.measure


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                #levelset_filename = os.path.splitext(os.path.join(self.data_source, 'SurfaceSamples', self.npyfiles[new_idx]))[0] + '.ply'
                #normalization_filename = os.path.join(self.data_source, 'NormalizationParameters', self.npyfiles[new_idx])
                if not os.path.isfile(
                    os.path.join(data_source, 'SdfSamples', instance_filename)
                ):
                    continue
                    
                if not os.path.isfile(
                    os.path.join(data_source, 'NormalizationParameters', instance_filename)):
                    continue
                if not os.path.isfile(
                    os.path.join(data_source, 'SurfaceSamples', os.path.join(dataset, class_name, instance_name + ".ply"))):
                    continue
                
                npzfiles += [instance_filename]
    print(f"Found {len(npzfiles)} files.")
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]

def load_levelset(levelset_filename, normalization_filename):    
    normalization_params = np.load(normalization_filename)
    
    unormalized = torch.FloatTensor(trimesh.load(levelset_filename).vertices)
    levelset_tensor = (unormalized + normalization_params['offset']) * normalization_params['scale']# Old version, works
    # levelset_tensor = unormalized * normalization_params['scale'] + normalization_params['offset']
    return levelset_tensor

def load_partial(partial_filename, normalization_filename):
    normalization_params = np.load(normalization_filename)
    unormalized = torch.FloatTensor(trimesh.load(partial_filename).vertices)
    partial_tensor = (unormalized + normalization_params['offset']) * normalization_params['scale']
    return partial_tensor

def read_sdf_samples_into_ram(sdf_filename, levelset_filename, partial_filename, normalization_params_filename, context_mode):
    sdf_npz = np.load(sdf_filename, allow_pickle=True)
    pos_tensor = remove_nans(torch.from_numpy(sdf_npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(sdf_npz["neg"]))
    
    levelset_tensor = torch.zeros((1,))
    partial_tensor = torch.zeros((1,))
    
    if context_mode == 'levelset':
        levelset_tensor = load_levelset(levelset_filename, normalization_params_filename)
    if context_mode == 'partial':
        partial_tensor = load_partial(partial_filename, normalization_params_filename)
    
    return [pos_tensor, neg_tensor, levelset_tensor, partial_tensor]


def unpack_sdf_samples(sdf_filename, levelset_filename, partial_filename, normalization_filename, subsampleSDF, subsampleLevelset, context_mode):
    npz = np.load(sdf_filename)

    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsampleSDF / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)
    
    if context_mode == 'levelset':
        levelset_tensor = load_levelset(levelset_filename, normalization_filename)
        context_idcs = np.random.choice(levelset_tensor.shape[0], subsampleLevelset, replace=False)
        levelset_points = levelset_tensor[context_idcs]
    elif context_mode == 'partial':
        levelset_tensor = load_partial(partial_filename, normalization_filename)
        context_idcs = np.random.choice(levelset_tensor.shape[0], subsampleLevelset, replace=True)
        levelset_points = levelset_tensor[context_idcs]
    else:
        levelset_points = torch.zeros((1,))
        
    return {'sdf': samples, 'levelset': levelset_points}


def unpack_sdf_samples_from_ram(data, subsampleSDF, subsampleLevelset):
    pos_tensor = data[0]
    neg_tensor = data[1]
    levelset_tensor = data[2]
    partial_tensor = data[3]
    
    levelset_points = levelset_tensor
    # Subsample Levelset
    context_idcs = np.random.choice(levelset_tensor.shape[0], subsampleLevelset, replace=True)
    levelset_points = levelset_tensor[context_idcs]
    
    if not subsampleSDF:
        subsampleSDF = pos_tensor.shape[0]

    # split the sample into half
    half = int(subsampleSDF / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    #return samples

    return {'sdf': samples, 'levelset': levelset_points, 'partial': partial_tensor}


class LevelsetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsampleSDF,
        subsampleLevelset,
        context_mode,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsampleSDF = subsampleSDF
        self.subsampleLevelset = subsampleLevelset
        self.context_mode = context_mode

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)
    
    def get_filenames(self, new_idx):
        instance_name = os.path.splitext(self.npyfiles[new_idx])[0]#.split('/')[-1]
        sdf_filename = os.path.join(self.data_source, 'SdfSamples', self.npyfiles[new_idx])
        levelset_filename = os.path.splitext(os.path.join(self.data_source, 'SurfaceSamples', self.npyfiles[new_idx]))[0] + '.ply'
        normalization_filename = os.path.join(self.data_source, 'NormalizationParameters', self.npyfiles[new_idx])
        partial_filename = os.path.join('/home/ericryanchan/depth_maps', instance_name, 'world_coords.ply')
        return sdf_filename, levelset_filename, partial_filename, normalization_filename

    def __getitem__(self, idx):        
        new_idx = idx
        while True:
            sdf_filename, levelset_filename, partial_filename, normalization_filename = self.get_filenames(new_idx)
            try:
                return unpack_sdf_samples(sdf_filename, levelset_filename, partial_filename, normalization_filename, self.subsampleSDF, self.subsampleLevelset, self.context_mode), new_idx
            except (FileNotFoundError, ValueError) as e:
                #print(e)
                new_idx = (new_idx + 1) % len(self)
                
                
def meta_split(sdf_tensor, levelset_tensor, context_mode):    
    if context_mode == 'levelset':
        xyz = sdf_tensor[:, :, 0:3]
        sdf_gt = sdf_tensor[:, :, 3:4]
        # Use levelset points, 0's as context; full sdf data as test
        meta_data = {'context':(levelset_tensor, torch.zeros(levelset_tensor.shape[0], levelset_tensor.shape[1], 1)),
                     'query':(xyz, sdf_gt)}
        
        return meta_data

    elif context_mode == 'dense':
        ######## Subsample half of the points as context
        context_inputs = []
        context_targets = []
        test_inputs = []
        test_targets = []
        
        batch_size = sdf_tensor.shape[0]
        for b in range(batch_size):
            idx = torch.randperm(sdf_tensor[b].shape[0]) # shuffle along points dimension
            sdf_tensor[b] = sdf_tensor[b][idx]

            context_length = sdf_tensor[b].shape[0]//2

            context_inputs.append(sdf_tensor[b][:context_length, :3])
            context_targets.append(sdf_tensor[b][:context_length, 3:])
            test_inputs.append(sdf_tensor[b][context_length:, :3])
            test_targets.append(sdf_tensor[b][context_length:, 3:])

        context_inputs = torch.stack(context_inputs, dim=0)
        context_targets = torch.stack(context_targets, dim=0)
        test_inputs = torch.stack(test_inputs, dim=0)
        test_targets = torch.stack(test_targets, dim=0)

        meta_data = {'context':(context_inputs, context_targets),
                     'query':(sdf_tensor[:,:,:3], sdf_tensor[:,:,3:])}
        return meta_data
        #########################
    elif context_mode == 'partial': # Same as levelset right now, just loading different points
        xyz = sdf_tensor[:, :, 0:3]
        sdf_gt = sdf_tensor[:, :, 3:4]
        # Use partial points, 0's as context; full sdf data as test
        meta_data = {'context':(levelset_tensor, torch.zeros(levelset_tensor.shape[0], levelset_tensor.shape[1], 1)),
                     'query':(xyz, sdf_gt)}
        
        return meta_data
    else:
        raise NotImplementedError
        
def create_samples(N=256, max_batch = 32768, offset=None, scale=None):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False
                   
    return samples

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
