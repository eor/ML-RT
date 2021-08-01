import numpy as np
import os.path as osp
import heapq
try:
    from common.utils import utils_load_config
except ImportError:
    from utils import utils_load_config

try:
    from plot import *
except ImportError:
    from common.plot import *


# -----------------------------------------------------------------
# Purpose of the functions below is to automatically run the data
# products through the plotting routines (after a training run) and
# and make sure the plots are placed in the corresponding 'plots'
# directories.
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# hard-coded parameters (for now)
# -----------------------------------------------------------------
DATA_PRODUCTS_DIR = 'data_products'
PLOT_DIR = 'plots'
PLOT_FILE_TYPE = 'pdf'  # or 'png'


# -----------------------------------------------------------------
#  setup for the loss function plots
# -----------------------------------------------------------------
def analysis_loss_plot(config):

    """
    function to load and plot the training and validation loss data

    Args:

        config object that is generated from user arguments when
        the main script is run
    """

    data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
    plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    train_loss_file = 'train_loss_%s_%d_epochs.npy'%(config.profile_type, config.n_epochs)
    val_loss_file = 'val_loss_%s_%d_epochs.npy'%(config.profile_type, config.n_epochs)

    train_loss = np.load(osp.join(data_dir_path, train_loss_file))
    val_loss = np.load(osp.join(data_dir_path, val_loss_file))

    plot_loss_function(
        lf1=train_loss,
        lf2=val_loss,
        epoch=config.n_epochs,
        lr=config.lr,
        output_dir=plot_dir_path,
        profile_type=config.profile_type,
        file_type=PLOT_FILE_TYPE
    )


# -----------------------------------------------------------------
# Automatically plot test profiles
# -----------------------------------------------------------------
def analysis_auto_plot_profiles(config, k=5, prefix='test'):

    # 1. read data
    data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
    plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    if prefix == 'test':
        epoch = config.n_epochs
    elif prefix == 'best':
        epoch = config.best_epoch

    parameter_true_file = prefix+'_parameters_%s_%d_epochs.npy'%(config.profile_type, epoch)
    profiles_true_file = prefix+'_profiles_true_%s_%d_epochs.npy'%(config.profile_type, epoch)
    profiles_gen_file = prefix+'_profiles_gen_%s_%d_epochs.npy'%(config.profile_type, epoch)

    parameters = np.load(osp.join(data_dir_path, parameter_true_file))
    profiles_true = np.load(osp.join(data_dir_path, profiles_true_file))
    profiles_gen = np.load(osp.join(data_dir_path, profiles_gen_file))

    # 2. compute MSE
    mse_array = (np.square((profiles_true) - (profiles_gen))).mean(axis=1)
    mse_array = np.log10(mse_array + 1e-11)

    # 3. find k lowest and largest MSE values and their respective indexes
    k_large_list = heapq.nlargest(k, range(len(mse_array)), mse_array.take)
    k_small_list = heapq.nsmallest(k, range(len(mse_array)), mse_array.take)

    # 4.  plot profiles for largest MSE
    print('Producing profile plot(s) for profiles with %d largest MSE'%(k))
    for i in range(len(k_large_list)):
        index = k_large_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        tmp_parameters = parameters[index]
        tmp_profile_true = profiles_true[index]
        tmp_profile_gen = profiles_gen[index]

        plot_profile_single(
            profile_true = tmp_profile_true, 
            profile_inferred = tmp_profile_gen, 
            n_epoch = epoch,
            output_dir =  plot_dir_path, 
            profile_type = config.profile_type,
            prefix = prefix,  
            parameters = tmp_parameters
        )

    # 5.  plot profiles for smallest MSE
    print('Producing profile plot(s) for profiles with %d smallest MSE'%(k))
    for i in range(len(k_small_list)):
        index = k_small_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        tmp_parameters = parameters[index]
        tmp_profile_true = profiles_true[index]
        tmp_profile_gen = profiles_gen[index]

        plot_profile_single(
            profile_true = tmp_profile_true, 
            profile_inferred = tmp_profile_gen, 
            n_epoch = epoch,
            output_dir =  plot_dir_path, 
            profile_type = config.profile_type,
            prefix = prefix, 
            parameters = tmp_parameters
        )


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there! Let\'s analyse some results\n')

    # F: local example
    base = '../test/run_2021_08_01__17_39_48'
    config = utils_load_config(base)
    # lr = 0.001
    k = 5
    profile = config.profile_type
    analysis_auto_plot_profiles(config, k=k, prefix='test')

    print('\n Completed! \n')