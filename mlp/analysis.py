from plot import *
import numpy as np
import os.path as osp
import heapq

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
def analysis_loss_plot(base_dir, epoch, lr, profile_choice='T'):

    """
    function to load and plot the training and validation loss data

    Args:
        base_dir: run directory
        epoch:
        lr: learning rate
        profile_choice: type of profiles used
    """

    data_dir_path = osp.join(base_dir, DATA_PRODUCTS_DIR)
    plot_dir_path = osp.join(base_dir, PLOT_DIR)

    train_loss_file = 'train_loss_%s_%d_epochs.npy'%(profile_choice, epoch)
    val_loss_file = 'val_loss_%s_%d_epochs.npy'%(profile_choice, epoch)

    train_loss = np.load(osp.join(data_dir_path, train_loss_file))
    val_loss = np.load(osp.join(data_dir_path, val_loss_file))

    plot_loss_function(
        lf1=train_loss,
        lf2=val_loss,
        epoch=epoch,
        lr=lr,
        output_dir=plot_dir_path,
        profile_type=profile_choice,
        file_type=PLOT_FILE_TYPE
    )


# -----------------------------------------------------------------
#  Automatically plot test profiles
# -----------------------------------------------------------------
# def analysis_auto_plot_profiles(base_dir, k, epoch, profile_choice='T', prefix='test'):
#
#     # 1. read data ... same as for the parameter-MSE plot
#     data_dir_path = osp.join(base_dir, DATA_PRODUCTS_DIR)
#     plot_dir_path = osp.join(base_dir, PLOT_DIR)
#
#     parameter_true_file = prefix+'_parameters_%s_%d_epochs.npy'%(profile_choice, epoch)
#     profiles_true_file = prefix+'_profiles_true_%s_%d_epochs.npy'%(profile_choice, epoch)
#     profiles_gen_file = prefix+'_profiles_gen_%s_%d_epochs.npy'%(profile_choice, epoch)
#
#     parameters = np.load(osp.join(data_dir_path, parameter_true_file))
#     profiles_true = np.load(osp.join(data_dir_path, profiles_true_file))
#     profiles_gen = np.load(osp.join(data_dir_path, profiles_gen_file))
#
#     # 2. compute MSE
#     mseArray = (np.square((profiles_true) - (profiles_gen))).mean(axis=1)
#     mseArray = np.log10(mseArray + 1e-11)
#
#     # 3. find k lowest and largest MSE values and their respective indexes
#     kLargeList = heapq.nlargest(k, range(len(mseArray)), mseArray.take)
#     kSmallList = heapq.nsmallest(k, range(len(mseArray)), mseArray.take)
#
#     # 4.  plot profiles for largest MSE
#     print('Producing profile plot(s) for profiles with %d largest MSE'%k)
#     for i in range(len(kLargeList)):
#         index = kLargeList[i]
#         print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mseArray[index], parameters[index]))
#
#         tmp_parameters = parameters[index]
#         tmp_profile_true = profiles_true[index]
#         tmp_profile_gen = profiles_gen[index]
#
#         plot_test_profiles(tmp_profile_true, tmp_profile_gen, epoch, plot_dir_path, profile_choice, tmp_parameters)
#
#     # 4.  plot profiles for smallest MSE
#     print('Producing profile plot(s) for profiles with %d smallest MSE'%k)
#     for i in range(len(kSmallList)):
#         index = kSmallList[i]
#         print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mseArray[index], parameters[index]))
#
#         tmp_parameters = parameters[index]
#         tmp_profile_true = profiles_true[index]
#         tmp_profile_gen = profiles_gen[index]
#
#         plot_test_profiles(tmp_profile_true, tmp_profile_gen, epoch, plot_dir_path, profile_choice, tmp_parameters)


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there! Let\'s analyse some results\n')


    # base = './output/run_20210520_224447'
    # lr = 0.001
    # epoch = 2000
    # k = 4
    # profile = 'T'
    # analysis_loss_plot(base, epoch, lr, profile)
    # analysis_auto_plot_profiles(base, k, epoch, profile_choice=profile, prefix='test')