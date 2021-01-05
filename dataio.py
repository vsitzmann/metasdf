import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.datasets.utils import download_file_from_google_drive
from torchmeta.datasets.utils import get_asset

import matplotlib.pyplot as plt

import numpy as np
#import trimesh
from pathlib import Path
import pickle
import os

import numpy as np
from PIL import Image
import os
import io
import json
import h5py

from tqdm.autonotebook import tqdm

import scipy
from torchmeta.utils.data import Task, MetaDataset
import math
from skimage import measure


def get_mgrid(sidelen):
    # Generate 2D pixel coordinates.
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen    
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords


def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)


class MNIST(torch.utils.data.Dataset):
    def __init__(self, split, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 transform=None, target_transform=None):
        super().__init__()

        assert split in ['train', 'test', 'val'], "Unknown split"

        self.mnist = torchvision.datasets.MNIST('./datasets/MNIST', train=True if split in ['train', 'val'] else False,
                                                download=True, transform=transform, target_transform=target_transform)
        
        # filter by selected numbers
        idx = [(x in selected_digits) for x in self.mnist.targets]
        self.mnist.targets = self.mnist.targets[idx]
        self.mnist.data = self.mnist.data[idx]
        
        # Take 10% of training dataset and create a validation dataset
        if split in ['train', 'val']:
            # Split into train and val splits
            torch.manual_seed(0)
            num_train = int(0.9 * len(self.mnist))
            num_val = len(self.mnist) - num_train
            train_dataset, val_dataset = torch.utils.data.random_split(self.mnist, [num_train, num_val])
            self.mnist = train_dataset if split=='train' else val_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx]


class MultiMNIST(torch.utils.data.Dataset):
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, split, selected_classes,
                 folder, gdrive_id, zip_filename, zip_md5, image_folder):
        self.gdrive_id = gdrive_id
        self.image_folder = image_folder
        self.folder = folder
        self.zip_md5 = zip_md5
        self.zip_filename = zip_filename

        self.root = os.path.join('./datasets/', self.folder)
        self.split_filename = os.path.join(self.root,
                                           self.filename.format(split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(split))

        self.download()

        self.data_file = h5py.File(self.split_filename, 'r')
        self.data = self.data_file['datasets']

        if selected_classes:
            self.selected_classes = selected_classes
        else:
            with open(self.split_filename_labels, 'r') as f:
                self.selected_classes = json.load(f)

        self.concat_data, self.labels = list(), list()
        for int_class, string_class in enumerate(self.selected_classes):
            sub_dataset = self.data[string_class]
            self.concat_data.append(sub_dataset)
            self.labels.extend([int_class]*len(sub_dataset))

        self.dataset = torch.utils.data.ConcatDataset(self.concat_data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.dataset[index]))
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels))

    def download(self):
        import zipfile
        import shutil
        import glob

        if self._check_integrity():
            return

        zip_filename = os.path.join(self.root, self.zip_filename)
        if not os.path.isfile(zip_filename):
            download_file_from_google_drive(self.gdrive_id, self.root,
                                            self.zip_filename, md5=self.zip_md5)

        zip_foldername = os.path.join(self.root, self.image_folder)
        if not os.path.isdir(zip_foldername):
            with zipfile.ZipFile(zip_filename, 'r') as f:
                for member in tqdm(f.infolist(), desc='Extracting '):
                    try:
                        f.extract(member, self.root)
                    except zipfile.BadZipFile:
                        print('Error: Zip file is corrupted')

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root,
                                           self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            image_folder = os.path.join(zip_foldername, split)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label,
                                                    '*.png'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),),
                                                   dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        if os.path.isdir(zip_foldername):
            shutil.rmtree(zip_foldername)


class DoubleMNIST(MultiMNIST):
    def __init__(self, split, selected_classes=None):
        super().__init__(split=split,
                         selected_classes=selected_classes,
                         folder='doublemnist',
                         gdrive_id='1MqQCdLt9TVE3joAMw4FwJp_B8F-htrAo',
                         zip_filename='double_mnist_seed_123_image_size_64_64.zip',
                         zip_md5='6d8b185c0cde155eb39d0e3615ab4f23',
                         image_folder='double_mnist_seed_123_image_size_64_64')


class TripleMNIST(MultiMNIST):
    def __init__(self, split, selected_classes=None):
        super().__init__(split=split,
                         selected_classes=selected_classes,
                         folder='triplemnist',
                         gdrive_id='1xqyW289seXYaDSqD2jaBPMKVAAjPP9ee',
                         zip_filename='triple_mnist_seed_123_image_size_84_84.zip',
                         zip_md5='9508b047f9fbb834c02bc13ef44245da',
                         image_folder='triple_mnist_seed_123_image_size_84_84')


class SignedDistanceTransform:
    def __call__(self, img_tensor):
        # Threshold.
        img_tensor[img_tensor<0.5] = 0.
        img_tensor[img_tensor>=0.5] = 1.

        # Compute signed distances with distance transform
        img_tensor = img_tensor.numpy()

        neg_distances = scipy.ndimage.morphology.distance_transform_edt(img_tensor)
        sd_img = img_tensor - 1.
        sd_img = sd_img.astype(np.uint8)
        signed_distances = scipy.ndimage.morphology.distance_transform_edt(sd_img) - neg_distances
        signed_distances /= float(img_tensor.shape[1])
        signed_distances = torch.Tensor(signed_distances)

        return signed_distances, torch.Tensor(img_tensor)


class SignedDistanceWrapper(torch.utils.data.Dataset):
    def __init__(self, img_dataset, size=(256,256), level_set_points=512):
        self.transform = Compose([
            Resize(size),
            ToTensor(),
            SignedDistanceTransform(),
        ])
        self.img_dataset = img_dataset
        self.meshgrid = get_mgrid(size[0])
        self.im_size = size
        self.level_set_points = level_set_points

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img, digit_class = self.img_dataset[item]

        signed_distance_img, binary_image = self.transform(img)
        signed_distance_img = signed_distance_img.reshape((-1, 1))

        # Compute the level set
        binary_image -= 0.5
        level_set = np.concatenate(measure.find_contours(binary_image.view(self.im_size).numpy(), 0.4), axis=0)
        level_set = torch.Tensor(level_set)
        level_set /= self.im_size[0]
        level_set -= 0.5

        num_random_points = self.level_set_points - level_set.shape[0]
        random_points = torch.zeros(num_random_points, 2)

        level_set_gt = torch.zeros(level_set.shape[0])
        random_points_gt = torch.ones(num_random_points) * -1.

        all_points = torch.cat((level_set, random_points), dim=0)
        all_gt = torch.cat((level_set_gt, random_points_gt), dim=0)[..., None]

        observation_dict = {'idx':item, 'coords':self.meshgrid, 'img':binary_image, 'level_set':all_points}
        gt_dict = {'sds':signed_distance_img, 'digit_class':digit_class, 'img':binary_image, 'ls_sds':all_gt}

        return observation_dict, gt_dict


def pack_dataset(dataset, path, max_length=float('Inf')):
    """
    Save a dataset to a pickled list. Specify max_length to a number e.g. 10 if
    you only want a portion of the dataset.
    """
    data = []
    with tqdm(total=min(max_length, len(dataset)), desc='Buffering Dataset') as pbar:
        for i, d in enumerate(dataset):
            if i >= max_length:
                break
            pbar.update(1)
            data.append(d)
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

        
class BufferDataset(torch.utils.data.Dataset):
    """
    Load a saved dataset from a filepath.
    
    max_length gives the maximum items to buffer--e.g. 1000 if you only want the first 1000 MNIST digits.
    Leave off if you want all 60000.
    
    force_reload=True deletes the dataset and regenerates it, useful if you've changed the dataset. Otherwise,
    loads dataset from saved file.
    """
    def __init__(self, dataset, max_length=float('Inf'), force_reload=False, save_path=None, filename=None):
        super().__init__()

        if not save_path:
            max_len_str = '' if max_length == float('Inf') else '-' + str(max_length)

            if not filename:
                filename = f'{dataset.__class__.__name__}{max_len_str}'

            save_path = f'datasets/{filename}'
        
        if force_reload and os.path.exists(save_path):
            os.remove(save_path)
        
        if not os.path.exists(save_path):
            pack_dataset(dataset, save_path, max_length)
        
        with open(save_path, 'rb') as f:
            self.data = pickle.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MetaSplitter(torch.utils.data.Dataset):
    def __init__(self, metadataset, dense=True, mode='train', num_samples=64**2//2):
        self.dataset = metadataset
        self.dense = dense
        self.mode = mode
        self.num_samples = num_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]

        if self.dense:
            context_tuple = sample[0]['coords'], sample[1]['sds']
        else:
            context_tuple = sample[0]['level_set'], sample[1]['ls_sds']

        if self.mode == 'train':
            # If training mode, we only sample num_samples inputs to condition on.
            all_inputs_targets = torch.cat(context_tuple, dim=1)

            context_idcs = np.random.choice(all_inputs_targets.shape[0], self.num_samples, replace=True)
            sampled_inputs_targets = all_inputs_targets[context_idcs]

            context_inputs = sampled_inputs_targets[:, :2]
            context_targets = sampled_inputs_targets[:, 2:]

            context_tuple = context_inputs, context_targets

        return {'train':context_tuple, 'test':(sample[0]['coords'], sample[1]['sds'], sample[1]['digit_class'])}
