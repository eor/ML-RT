import os
import os.path as osp
import numpy as np
import random
from torch.utils.data import DataLoader

import sys; sys.path.append('..')
from common.utils import *
from common.settings import *
from common.filter import *
from common.dataset import RTdata
import common.settings_parameters as ps


def run_set_parameter_limits(config):
    """
    Adds the global parameter limits and latex names to the config object

    Args:
        config: user config object

    Returns:
        user config object

    """

    if config.n_parameters == 5:
        config.parameter_limits = ps.p5_limits
        config.parameter_names_latex = ps.p5_names_latex

    if config.n_parameters == 8:
        config.parameter_limits = ps.p8_limits
        config.parameter_names_latex = ps.p8_names_latex

    return config


def run_setup_run(config):
    """
    Sets up unique run directory and saves config file

    Args:
        config: user config object

    Returns:
        data products path

    """

    run_id = 'run_' + utils_get_current_timestamp()
    config.out_dir = os.path.join(config.out_dir, run_id)

    utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)

    return data_products_path


def run_get_data_loaders_one_profile(config):
    """
    Loads data, filter if necessary, and return data loaders containing one profile

    Args:
        config: user config object

    Returns:
        three data loaders (training validation, testing )

    """

    # -----------------------------------------------------------------
    # Check if data files exist / read data and shuffle / rescale parameters
    # -----------------------------------------------------------------
    H_profile_file_path = utils_join_path(config.data_dir, H_II_PROFILE_FILE)
    T_profile_file_path = utils_join_path(config.data_dir, T_PROFILE_FILE)
    global_parameter_file_path = utils_join_path(config.data_dir, GLOBAL_PARAMETER_FILE)

    H_profiles = np.load(H_profile_file_path)
    T_profiles = np.load(T_profile_file_path)
    global_parameters = np.load(global_parameter_file_path)

    # -----------------------------------------------------------------
    # OPTIONAL: Filter (blow-out) profiles
    # -----------------------------------------------------------------
    if config.filter_blowouts:
        (H_profiles,
         T_profiles,
         global_parameters) = filter_blowout_profiles(H_profiles, T_profiles, global_parameters)

    if config.filter_parameters:
        (global_parameters,
         [H_profiles, T_profiles]) = filter_cut_parameter_space(global_parameters, [H_profiles, T_profiles])

    # -----------------------------------------------------------------
    # log space?
    # -----------------------------------------------------------------
    if USE_LOG_PROFILES:
        H_profiles = np.log10(H_profiles + 1.0e-6)  # add a small number to avoid trouble
        T_profiles = np.log10(T_profiles)

    # -----------------------------------------------------------------
    # shuffle / rescale parameters
    # -----------------------------------------------------------------
    if SCALE_PARAMETERS:
        global_parameters = utils_scale_parameters(limits=config.parameter_limits, parameters=global_parameters)

    if SHUFFLE:
        np.random.seed(SHUFFLE_SEED)
        n_samples = H_profiles.shape[0]
        indices = np.arange(n_samples, dtype=np.int32)
        indices = np.random.permutation(indices)
        H_profiles = H_profiles[indices]
        T_profiles = T_profiles[indices]
        global_parameters = global_parameters[indices]

    # -----------------------------------------------------------------
    # we are doing one profile at a time
    # -----------------------------------------------------------------
    if config.profile_type == 'H':
        profiles = H_profiles
    else:
        profiles = T_profiles

    # -----------------------------------------------------------------
    # data loaders
    # -----------------------------------------------------------------
    training_data = RTdata(profiles, global_parameters, split='train', split_frac=SPLIT_FRACTION)
    validation_data = RTdata(profiles, global_parameters, split='val', split_frac=SPLIT_FRACTION)
    testing_data = RTdata(profiles, global_parameters, split='test', split_frac=SPLIT_FRACTION)

    train_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=config.batch_size)
    test_loader = DataLoader(testing_data, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader


def run_get_data_loaders_four_profiles(config):
    """
    Loads data, filter if necessary, and return data loaders containing all four profiles

    Args:
        config: user config object

    Returns:
        three data loaders (training validation, testing )

    """

    # -----------------------------------------------------------------
    # Check if data files exist / read data and shuffle / rescale parameters
    # -----------------------------------------------------------------
    H_II_profile_file_path = utils_join_path(config.data_dir, H_II_PROFILE_FILE)
    T_profile_file_path = utils_join_path(config.data_dir, T_PROFILE_FILE)
    He_II_profile_file_path = utils_join_path(config.data_dir, He_II_PROFILE_FILE)
    He_III_profile_file_path = utils_join_path(config.data_dir, He_III_PROFILE_FILE)

    global_parameter_file_path = utils_join_path(config.data_dir, GLOBAL_PARAMETER_FILE)

    H_II_profiles = np.load(H_II_profile_file_path)
    T_profiles = np.load(T_profile_file_path)
    He_II_profiles = np.load(He_II_profile_file_path)
    He_III_profiles = np.load(He_III_profile_file_path)
    global_parameters = np.load(global_parameter_file_path)

    # -----------------------------------------------------------------
    # OPTIONAL: Filter (blow-out) profiles
    # -----------------------------------------------------------------
    if config.filter_blowouts:
        (H_II_profiles,
         T_profiles,
         He_II_profiles,
         He_III_profiles,
         global_parameters) = filter_blowout_profiles(H_II_profiles=H_II_profiles,
                                                      T_profiles=T_profiles,
                                                      global_parameters=global_parameters,
                                                      He_II_profiles=He_II_profiles,
                                                      He_III_profiles=He_III_profiles
                                                      )

    if config.filter_parameters:
        (global_parameters,
         [H_II_profiles,
          T_profiles,
          He_II_profiles,
          He_III_profiles]) = filter_cut_parameter_space(global_parameters=global_parameters,
                                                         profiles=[H_II_profiles,
                                                                   T_profiles,
                                                                   He_II_profiles,
                                                                   He_III_profiles]
                                                         )

    # -----------------------------------------------------------------
    # log space?
    # -----------------------------------------------------------------
    if USE_LOG_PROFILES:
        # add a small number to avoid trouble
        H_II_profiles = np.log10(H_II_profiles + 1.0e-6)
        He_II_profiles = np.log10(He_II_profiles + 1.0e-6)
        He_III_profiles = np.log10(He_III_profiles + 1.0e-6)
        T_profiles = np.log10(T_profiles)

    # -----------------------------------------------------------------
    # shuffle / rescale parameters
    # -----------------------------------------------------------------
    if SCALE_PARAMETERS:
        global_parameters = utils_scale_parameters(limits=config.parameter_limits, parameters=global_parameters)

    if SHUFFLE:
        np.random.seed(SHUFFLE_SEED)
        n_samples = H_II_profiles.shape[0]
        indices = np.arange(n_samples, dtype=np.int32)
        indices = np.random.permutation(indices)

        H_II_profiles = H_II_profiles[indices]
        T_profiles = T_profiles[indices]
        He_II_profiles = He_II_profiles[indices]
        He_III_profiles = He_III_profiles[indices]
        global_parameters = global_parameters[indices]

    profiles = np.stack((H_II_profiles, T_profiles, He_II_profiles, He_III_profiles), axis=1)

    # -----------------------------------------------------------------
    # data loaders
    # -----------------------------------------------------------------
    training_data = RTdata(profiles, global_parameters, split='train', split_frac=SPLIT_FRACTION)

    validation_data = RTdata(profiles, global_parameters, split='val', split_frac=SPLIT_FRACTION)

    testing_data = RTdata(profiles, global_parameters,  split='test', split_frac=SPLIT_FRACTION)

    train_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=config.batch_size)
    test_loader = DataLoader(testing_data, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader