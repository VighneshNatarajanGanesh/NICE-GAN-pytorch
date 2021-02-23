import torch.utils.data as data

from PIL import Image

import numpy as np

import h5py

import os
import os.path

HDF5_DATASET_NAME = 'hs_data'

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


# change 1: added .hdf5 as valid input format.
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.hdf5']


def pil_and_hdf5_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # for hdf5
    if(path.endswith('.hdf5')):
        with h5py.File(path, 'r') as f:
            data = f[HDF5_DATASET_NAME][:].astype(np.float32)
            # normalize it to 0 to 1
            data /= (data.max() - data.min() + 0.0001)
            # normalize to -1 to 1
            data *= 2
            data -= 1
            
            # the torrchvision.transforms.toTensor rescales input to the range -1 to 1 in certain conditions,
            # we want to scale -1 to 1
            # so scale in the dataloader itself!
            # link: https://pytorch.org/docs/stable/torchvision/transforms.html
            
            return data

            # note:
            # DONOT USE: np.array(f[hdf5_dataset_name]) it was much slower in testing
            
    # for other types
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            data = np.array(img.convert('RGB')).astype(np.float32)
            
            # here too we want the scaling to be from -1 to 1
            # the to tensor normalizes 0 to 1 only if the numpy array is of type uint8
            # so return float32 image instead
            # link: https://pytorch.org/docs/stable/torchvision/transforms.html
            
            # normalize it to 0 to 1
            data /= (data.max() - data.min() + 0.0001)
            # normalize to -1 to 1
            data *= 2
            data -= 1
            
            return data


def default_loader(path):
    return pil_and_hdf5_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
