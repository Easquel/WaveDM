import torch
import shutil
import os
import torchvision.utils as tvu

from collections import OrderedDict


def save_image(img, file_directory,normalize=False):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory, normalize=normalize)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path, map_location='cpu')
    else:
        print("load to this device:",device)

        checkpoints = torch.load(path, map_location=device)

        return checkpoints


