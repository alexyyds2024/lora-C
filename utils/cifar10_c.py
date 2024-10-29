import os
import hashlib
import numpy as np

from PIL import Image
import torch.utils.data as data

# Corrupted CIFAR-10/100 from https://github.com/hendrycks/robustness
# Data downloaded from https://drive.google.com/drive/folders/1HDVw6CmX3HiG0ODFtI75iIfBDxSiSz2K
# 19 different corruption types, in each type, 5 different severity
# organized as follows [severity 1: 1-10,000; severity 2: 1-10,000, ...,
# severity 5: 1-10,000].
class CorruptedCIFAR(data.Dataset):

    corruption_types = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
        'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
        'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter',
        'speckle_noise', 'zoom_blur'
    ]

    def __init__(self, root, corruption_type=None,
                 transform=None, target_transform=None):
        self.corrupted_data_root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.test_data = []
        self.test_labels = []
        if corruption_type is None:
            for c_type in self.corruption_types:
                self.test_data.append(np.load(os.path.join(self.corrupted_data_root,
                                                           '{}.npy'.format(c_type))))
                self.test_labels += np.load(os.path.join(self.corrupted_data_root,
                                                         'labels.npy')).tolist()
        else:
            self.test_data.append(np.load(os.path.join(self.corrupted_data_root,
                                                       '{}.npy'.format(corruption_type))))
            self.test_labels += np.load(os.path.join(self.corrupted_data_root,
                                                     'labels.npy')).tolist()

        self.test_data = np.concatenate(self.test_data, axis=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.corrupted_data_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str