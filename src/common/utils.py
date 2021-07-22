import numpy as np
import os
import torch
import os.path as osp
from datetime import datetime

# -----------------------------------------------------------------
# functions to scale and re-scale parameters
# -----------------------------------------------------------------

# scale all parameters to [0,1]
def utils_scale_parameters(limits, parameters):

    for i in range(parameters.shape[1]):

        a = limits[i][0]
        b = limits[i][1]

        parameters[:,i] = (parameters[:,i] - a)/(b - a)

    return parameters


# re-scale _all_ parameters to original limits
def utils_rescale_parameters(limits, parameters):

    for i in range(parameters.shape[1]):

        a = limits[i][0]
        b = limits[i][1]

        parameters[:, i] =  parameters[:, i] * (b - a) + a

    return parameters


# re-scale a single given parameter
def utils_rescale_parameters_single(limits, p):

    for i in range(0, len(p)):

        a = limits[i][0]
        b = limits[i][1]

        p[i] = p[i] * (b - a) + a

    return p


# -----------------------------------------------------------------
# normalise / de-mean profile data
# -----------------------------------------------------------------
def utils_normalise_profiles(profiles):

    # for each profile we
    # 1. subtract the mean
    # 2. divide by the variance

    n_profiles = np.shape(profiles)[0]

    for i in range(0, n_profiles):

        mean = np.var(profiles[i])
        variance = np.var(profiles[i], ddof=1)

        profiles[i] -= mean
        # profiles[i] /= variance

    return profiles


# -----------------------------------------------------------------
# join path and check if file exists
# -----------------------------------------------------------------
def utils_join_path(directory, dataFile):

    a = osp.join(directory, dataFile)

    if not osp.exists(a):
        print('Error: File not found:\n\n  %s\n\nExiting.'%a)
        exit(1)

    return a


# -----------------------------------------------------------------
# create output directorie(s)
# -----------------------------------------------------------------
def utils_create_output_dirs(list_of_dirs):

    for x in list_of_dirs:
        if not osp.exists(x):
            os.makedirs(x)
            print('Created directory:\t%s'%x)


# -----------------------------------------------------------------
# 1st and 2nd derivatives of a given 1D array
# -----------------------------------------------------------------
def utils_derivative_1(array, norm=None, absolute=False, mode='torch'):

    # assumes 1D array
    if mode=='torch':
        derivatives = torch.zeros(array.shape)

        # 2d version  (batch x profiles)
        derivatives[:, 1:-1] = (array[:, 2:] - array[:, 0:-2]) / 2.
        derivatives[:, 0] = derivatives[:, 1]
        derivatives[:, -1] = derivatives[:, -2]

    else:
        derivatives = np.zeros(array.shape)

        derivatives[1:-1] = (array[2:] - array[0:-2]) / 2.
        derivatives[0] = derivatives[1]
        derivatives[-1] = derivatives[-2]

    if absolute:
        derivatives = abs(derivatives)

    # TODO: check if this still works
    if norm == 'max':
        derMax = (derivatives.max(axis=1)).reshape((derivatives.shape[0], 1))
        derMax[derMax == 0] = 1              # we should not divide by zero
        derivatives = derivatives / derMax   # normalization of derivative using the maximum value

    return derivatives


def utils_derivative_2(array, norm=None):
    if norm == 'max':
        return utils_derivative_1(utils_derivative_1(array), norm='max', absolute=True)
    else:
        return utils_derivative_1(utils_derivative_1(array), absolute=True)


# -----------------------------------------------------------------
# save state of model
# -----------------------------------------------------------------
def utils_save_model(state, path, profile_choice, n_epoch, best_model=True, file_name=None):

    # if no file name is provided, construct one here
    if file_name is None:
        file_name = 'model_%s_%d_epochs.pth.tar'%(profile_choice, n_epoch)

        if best_model:
            file_name = 'best_'+file_name

    path = osp.join(path, file_name)
    torch.save(state, path)
    print('Saved model to:\t%s' % path)


# -----------------------------------------------------------------
# save loss function as numpy object
# -----------------------------------------------------------------
def utils_save_loss(loss_array, path, profile_choice, n_epoch, prefix='train'):

    file_name = prefix + '_loss_%s_%d_epochs.npy'%(profile_choice, n_epoch)
    path = osp.join(path, file_name)
    np.save(path, loss_array)
    print('Saved %s loss function to:\t%s' %(prefix, path))


# -----------------------------------------------------------------
# save parameters and profiles (true & inference)
# -----------------------------------------------------------------
def utils_save_test_data(parameters, profiles_true, profiles_gen, path, profile_choice, epoch, prefix='test'):

    parameter_filename = prefix + '_parameters_%s_%d_epochs.npy'%(profile_choice, epoch+1)
    profiles_true_filename = prefix + '_profiles_true_%s_%d_epochs.npy'%(profile_choice, epoch+1)
    profiles_gen_filename = prefix + '_profiles_gen_%s_%d_epochs.npy' % (profile_choice, epoch+1)

    parameter_path = osp.join(path, parameter_filename)
    profiles_true_path = osp.join(path, profiles_true_filename)
    profiles_gen_path = osp.join(path, profiles_gen_filename)

    print('\nSaving results in the following files:\n')
    print('\t%s' % parameter_path)
    print('\t%s' % profiles_true_path)
    print('\t%s\n' % profiles_gen_path)

    np.save(parameter_path, parameters)
    np.save(profiles_true_path, profiles_true)
    np.save(profiles_gen_path, profiles_gen)


# -----------------------------------------------------------------
# Filter blow-out profiles, i.e. filter ionisation profiles whose
# ionisation front is beyond the edge of the computing grid and
# their corresponding temperature counterparts
# -----------------------------------------------------------------
def utils_filter_profiles(H_profiles, T_profiles, global_parameters ):

    threshold = 0.9
    # identify all blow-outs
    # blow out is a x_{H_{II}} profile for which all array values are above a certain threshold (e.g. < 0.9)

    # 0) for each profile find the min value
    profile_minima = np.min(H_profiles, axis=1)

    # 1) find all profiles for which the min > threshold
    deletion_indices = np.where(profile_minima>threshold)[0]

    for i in range(0, len(deletion_indices)):
        index = deletion_indices[i]

        #print("Deletions for index %d [%d]:"%(i, index))
        #print("Parameters:", global_parameters[index,:])
        #print("Hprofile:", H_profiles[index,:])
        #print("--------------------------------------------------------------------------------------")

    H_profiles = np.delete(H_profiles, deletion_indices, axis=0)
    T_profiles = np.delete(T_profiles, deletion_indices, axis=0)
    global_parameters = np.delete(global_parameters, deletion_indices, axis=0)

    print("\nBlow-out filter: Deleting a total of %d samples. %d remaining."%(len(deletion_indices), len(H_profiles)))

    return H_profiles, T_profiles, global_parameters


# -----------------------------------------------------------------
# Current time stamp as a string
# -----------------------------------------------------------------
def utils_get_current_timestamp():

    # return datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# -----------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------
def utils_create_run_directories(main_dir, data_products_dir='data_products', plot_dir='plots'):

    d = osp.join(main_dir, data_products_dir)
    p = osp.join(main_dir, plot_dir)

    print('\nCreating directories:\n')
    print('\t' + main_dir)
    print('\t' + d)
    print('\t' + p)

    os.makedirs(main_dir, exist_ok=False)
    os.makedirs(d, exist_ok=False)
    os.makedirs(p, exist_ok=False)


# -----------------------------------------------------------------
# Write config to file
# -----------------------------------------------------------------
def utils_write_config_to_file(config, filename='log_config'):

    dir = config.out_dir

    p = osp.join(dir, filename)

    print('\nWriting config to file:\n')
    print('\t' + p)

    with open(p, 'w') as f:
        for arg in vars(config):
            line = str(arg) +'\t' + str(getattr(config, arg)) + '\n'
            f.write(line)

        time = utils_get_current_timestamp()
        f.write('\ncurrent time stamp\t'+ time + '\n')


