from __future__ import print_function, division
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import logging
from libs.utils import to_one_hot
from PIL import Image
from skimage import exposure
import pandas as pd
import skimage.transform
from libs.utils import txt_to_list, generate_mask

logger = logging.getLogger(__name__)
VALID_EXTENSIONS = ('.jpg', '.png')
VALID_ENCODING = ('positional', 'random', 'constant', 'noencoding', 'positional_sine')


def is_valid_image(filename):
    return filename.lower().endswith(VALID_EXTENSIONS)


class CLAHE(object):
    def __call__(self, img: np.ndarray):
        if np.argmin(img.shape) == 0:
            img = np.moveaxis(img, -1, 0)
        return exposure.equalize_adapthist(img)


class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


class OCTSliceMaskDataset(Dataset):

    def __init__(self, mode, image_set="data/slices", srf_gt_filename='', irf_gt_filename='', transform_image=None,
                 split_ratio=None, encoding_type='positional', fold=None, folds_dir='', volumes_remove_file=None):
        '''
        labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
        '''
        assert (split_ratio is not None or fold is not None) or mode == 'all', \
            'Either split ratio or fold must be provided'

        try:
            assert encoding_type in VALID_ENCODING, f'Encoding type must be one of {", ".join(VALID_ENCODING)}. {encoding_type} was provided.'
        except:
            print('Positional encoding chosen')
            encoding_type = 'positional'
        self.mode = mode
        self.filenames = self._get_files_directory(image_set)
        self.centers = self.load_centers(image_set)

        if split_ratio is not None:
            self.filenames = self._split_files(image_set, self.filenames, self.mode, split_ratio)
        elif fold is not None:
            assert type(fold) == int, 'Fold must be an integer.'
            assert folds_dir != '', 'A folds directory must be provided'
            self.filenames = self._get_files_fold(folds_dir, self.filenames, self.mode, fold)
        elif volumes_remove_file is not None:
            with open(volumes_remove_file) as f:
                volumes_remove = f.read().splitlines()
            self.filenames = [file for file in self.filenames if file.split('/')[-2] not in volumes_remove]

        self.dataset_len = len(self.filenames)
        self.encoding_type = encoding_type

        self.gt_labels = self._get_ground_truth(srf_gt_filename, irf_gt_filename)
        self.loss_weights = self._get_loss_weights()

        self.transform_image = transform_image

    @staticmethod
    def load_centers(directory):
        data = np.load(directory.parents[0].joinpath('results_center_mod.npy'), allow_pickle=True)
        return data.item()

    @staticmethod
    def _get_files_directory(directory):
        files = []
        for root, folder, fnames in os.walk(directory, followlinks=True):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_image(path):
                    files.append(path)

        return files

    @staticmethod
    def _get_files_fold(directory, filenames, mode, fold):
        if mode == 'validation':
            folders = txt_to_list(str(directory) + f'/validation.txt')
        else:
            folders = txt_to_list(str(directory) + f'/{mode}_{fold}.txt')

        filenames = [file for file in filenames if file.split('/')[-2] in folders]
        return filenames

    @staticmethod
    def _split_files(directory, filenames, mode, split_ratio):
        if mode == 'train':
            try:
                folders = txt_to_list(str(directory) + '/train_{:.2f}.txt'.format(split_ratio))
            except FileNotFoundError:
                raise FileNotFoundError('Splitting file not found. Generate train and test and come back.')

            filenames = [file for file in filenames if file.split('/')[-2] in folders]

        elif mode == 'test':
            try:
                folders = txt_to_list(str(directory) + '/test_{:.2f}.txt'.format(split_ratio))
            except FileNotFoundError:
                raise FileNotFoundError('Splitting file not found. Generate train and test and come back.')

            filenames = [file for file in filenames if file.split('/')[-2] in folders]

        return filenames

    @staticmethod
    def _get_ground_truth(srf_gt_filename, irf_gt_filename):
        srf_gt = pd.read_csv(srf_gt_filename)
        irf_gt = pd.read_csv(irf_gt_filename)

        srf_gt = srf_gt[['image_name', 'bscan', 'present', 'mm1', 'mm3', 'mm6']]
        irf_gt = irf_gt[['image_name', 'bscan', 'present', 'mm1', 'mm3', 'mm6']]

        gt = srf_gt.merge(irf_gt, on=['image_name', 'bscan'], suffixes=('_srf', '_irf'), how='outer')
        gt = gt.set_index(['image_name', 'bscan'])
        gt.fillna(value=0, inplace=True)
        gt = gt.sort_index(level=[0, 1])
        gt['healthy'] = gt.sum(axis=1)
        gt.healthy = gt.healthy.apply(lambda x: 1 if x == 0 else 0)
        gt = gt[
            ['healthy', 'present_srf', 'present_irf', 'mm1_srf', 'mm3_srf', 'mm6_srf', 'mm1_irf', 'mm3_irf', 'mm6_irf']]

        # Drop nans in index
        c = gt.index.names
        gt = gt.reset_index().dropna().drop_duplicates().set_index(c)
        return gt

    def _get_loss_weights(self):
        labels_sum = self.gt_labels.sum(axis=0)
        largest_class = max(labels_sum)
        weights = (largest_class / labels_sum).to_numpy()
        weights = torch.from_numpy(weights)
        return weights

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.array(Image.open(self.filenames[idx]).convert('L')) / 255.
        image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=-1)  # 3 repetitions or it won't work with CLAHE
        image_name = self.filenames[idx].split('/')[-2]
        bscan_number = int(self.filenames[idx].split('_')[-1].split('.')[0])
        center = torch.tensor([self.centers[image_name]])

        if self.encoding_type == 'positional' or self.encoding_type == 'noencoding':
            slices = to_one_hot(bscan_number, (1, 49))[0]
        elif self.encoding_type == 'random':
            random_number = np.random.randint(0, 49)
            slices = to_one_hot(random_number, (1, 49))
        elif self.encoding_type == 'constant':
            number = 0
            slices = to_one_hot(number, (1, 49))
        elif self.encoding_type == 'positional_sine':
            slices = to_one_hot(bscan_number, (1, 49))

        if self.transform_image:
            image = self.transform_image(image)

        if self.encoding_type not in ['noencoding', 'positional_sine']:
            image = image[:2]  # We only take 2 because the embedding will be concatenated later

        # Healthy, SRF Present, IRF Present, SRF1, SRF3, SRF6, IRF1, IRF3, IRF6
        gt_image_label = self.gt_labels.loc[image_name, bscan_number].to_numpy()
        gt_image_label = torch.from_numpy(gt_image_label).squeeze(0)

        sample = {'images': image, 'slices': slices, 'labels': gt_image_label, 'image_name': image_name,
                  'slice_number': bscan_number, 'center': center}
        return sample


class SegmentationColumnDataset(Dataset):
    def __init__(self, image_root, label_root, sample_file=None, transform_image=None):
        self.image_root = image_root
        self.label_root = label_root
        self.sample_file = sample_file
        self.transform_image = transform_image

        self.images = self._get_files_directory(self.image_root)

        data = pd.read_csv(self.label_root)
        if sample_file is not None:
            with open(sample_file) as f:
                sample = f.read().splitlines()
            data = data[data.image_name.isin(sample)].copy()
            available_volumes = data.image_name.unique()
            self.images = [image for image in self.images if image.split('/')[-2] in available_volumes]
        self.labels = data
        self.num_columns_image = (len(self.labels.columns) - 2) // 2
        self._build_healthy_labels()
        self.labels = self.labels[self._build_column_ordering()]

    def __len__(self):
        return len(self.labels)

    def _build_healthy_labels(self):
        for i in range(self.num_columns_image):
            # NOT OR operation. If there is one of the two, it's not healthy
            self.labels[f'healthy_{i}'] = ~(self.labels[f'srf_{i}'].astype(bool) | self.labels[f'irf_{i}'].astype(bool))
            self.labels[f'healthy_{i}'] = self.labels[f'healthy_{i}'].astype(int)

    def _build_column_ordering(self):
        column_ordering = ['image_name', 'bscan']
        for i in range(self.num_columns_image):
            for b in ['healthy', 'srf', 'irf']:
                column_ordering.append(f'{b}_{i}')
        return column_ordering

    @staticmethod
    def _get_files_directory(directory):
        files = []
        for root, folder, fnames in os.walk(directory, followlinks=True):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_image(path):
                    files.append(path)

        return files

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('L')) / 255.
        image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=-1)  # 3 repetitions or it won't work with CLAHE
        image_name = self.images[idx].split('/')[-2]
        bscan_number = int(self.images[idx].split('_')[-1].split('.')[0])

        # Remove the first two columns because they are the image name and the bscan number
        label = self.labels[(self.labels.image_name == image_name) & (self.labels.bscan == bscan_number)].to_numpy()[0,
                2:]
        label = torch.from_numpy(label.astype(int))

        slices = to_one_hot(bscan_number, (1, 49))

        if self.transform_image:
            image = self.transform_image(image)

        center = torch.tensor([24])

        sample = {'images': image, 'labels': label, 'image_name': image_name, 'slice_number': bscan_number,
                  'slices': slices, 'center': center}
        return sample


if __name__ == '__main__':
    t = transforms.Compose([CLAHE(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # d = OCTSliceMaskDataset(mode='train', image_set='../../../Datasets/Location/extractedTifs',
    #                         srf_gt_filename='../../../Datasets/Location/BSCAN/srf_location.csv',
    #                         irf_gt_filename='../../../Datasets/Location/BSCAN/irf_location.csv',
    #                         transform_image=t, encoding_type='random')
    #
    # a = d[0]

    d = SegmentationColumnDataset(image_root='../../../Datasets/Location/extractedTifs',
                                  label_root='../../../Datasets/Segmentation/segmentation_columns.csv',
                                  sample_file='../../../Datasets/Segmentation/random_sample_reviewed.txt')
    a = d[0]
