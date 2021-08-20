import torch
from torch.utils.data import Dataset
import numpy as np

class RTdata(Dataset):

    def __init__(self, profile_data, parameter_data, derivative_data=None, split='train', split_frac=(0.8, 0.1, 0.1)):
        # -------------------------------------------------------------------------------------------------
        # Splits the given data into Train, Test and Validation set
        # -------------------------------------------------------------------------------------------------
        # Args:
        # profile_data: numpy array of shape (num_samples, num_profiles, profile_len) or
        #               (num_samples, profile_len) containing data for all the profiles
        # parameter_data: numpy array of shape (num_samples, profile_len) containing parameters
        #                 corresponding to all the profiles
        # split: whether to return data correspoding to 'train', 'test' or 'val' dataset
        # split_fraction: ratio's in which data needs to be splitted (must add upto 1)
        # --------------------------------------------------------------------------------------------------
        if len(profile_data.shape) != 3:
            profile_data = profile_data[:, np.newaxis, :]
        if derivative_data is not None and len(derivative_data.shape) != 3:
            derivative_data = derivative_data[:, np.newaxis, :]

        train_frac, val_frac, test_frac = split_frac

        if sum(split_frac) != 1:
            print('Error: Fractions of train | val | test should add up to 1.0')
            exit(1)

        n_samples = profile_data.shape[0]

        self.profiles = profile_data

        if split == 'train':
            begin = 0
            last = int(train_frac * n_samples)

        if split == 'val':
            begin = int(train_frac * n_samples)
            last = int((train_frac + val_frac) * n_samples)

        if split == 'test':
            begin = int((train_frac + val_frac) * n_samples)
            last = -1

        self.profiles = torch.from_numpy(self.profiles[begin:last]).type(torch.FloatTensor)
        self.parameters = torch.from_numpy(parameter_data[begin:last]).type(torch.FloatTensor)
        if derivative_data:
            self.derivatives = torch.from_numpy(derivative_data[begin:last]).type(torch.FloatTensor)

    def __len__(self):
        return self.profiles.shape[0]

    def __getitem__(self, index):
        return self.profiles[index], self.parameters[index]


class RTdataWithDerivatives(Dataset):

    def __init__(self, profile_data, parameter_data, derivative_data, split='train', split_frac=(0.8, 0.1, 0.1)):

        train_frac, val_frac, test_frac = split_frac

        if sum(split_frac) != 1:
            print('Error: Fractions of train | val | test should add up to 1.0')
            exit(1)

        n_samples = profile_data.shape[0]

        self.profiles = profile_data

        if split == 'train':
            begin = 0
            last = int(train_frac * n_samples)

        if split == 'val':
            begin = int(train_frac * n_samples)
            last = int((train_frac + val_frac) * n_samples)

        if split == 'test':
            begin = int((train_frac + val_frac) * n_samples)
            last = -1

        self.profiles = torch.from_numpy(self.profiles[begin:last]).type(torch.FloatTensor)
        self.parameters = torch.from_numpy(parameter_data[begin:last]).type(torch.FloatTensor)
        self.derivatives = torch.from_numpy(derivative_data[begin:last]).type(torch.FloatTensor)

    def __len__(self):
        return self.profiles.shape[0]

    def __getitem__(self, index):

        return self.profiles[index], self.parameters[index], self.derivatives[index]
