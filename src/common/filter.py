import numpy as np
try:
    from common.utils import utils_get_user_param_limits
except ImportError:
    from utils import utils_get_user_param_limits


# -----------------------------------------------------------------
# Cuts in the parameter space
# -----------------------------------------------------------------
def filter_cut_parameter_space(global_parameters, profiles, user_config_path=''):
    """
    This function makes cuts to the parameter space according to some user-specified limits.
    The limits could be provided as an argument (2D numpy array) in the same format as the ones in
    settings_parameters.py. They could be either come from a user_settings.py or we could use the hard
    coded parameter section in the script files (e.g. mlp.py).

     Args:
        global_parameters: numpy objects containing the parameters
        profiles: list of numpy objects containing profiles
        user_config_path:

     Returns:
         Filtered numpy objects
     """

    # choose limits to use based on number of parameters
    original_len, n_parameters = global_parameters.shape
    param_limits = utils_get_user_param_limits(user_config_path)
    limits = param_limits['p5_limits'] if n_parameters == 5 else param_limits['p8_limits']

    deletion_indices = []
    for i in range(len(limits)):
        lower_limit = limits[i][0]
        upper_limit = limits[i][1]

        original_parameters = global_parameters[:, i]
        # find all indices where parameters don't lie between required range for all parameters
        deletion_indices += (np.where((original_parameters < lower_limit) | (original_parameters > upper_limit))[0]).tolist()

    # delete the repeating indices from the list
    deletion_indices = list(set(deletion_indices))

    # delete the entries from dataset corresponding to the deletion_indices
    for i in range(len(profiles)):
        profiles[i] = np.delete(profiles[i], deletion_indices, axis=0)

    global_parameters = np.delete(global_parameters, deletion_indices, axis=0)

    print("\nParameter space filter: Deleting a total of {} samples. {} remaining.".format(len(deletion_indices),
                                                                                           len(global_parameters)))

    return global_parameters, profiles


# -----------------------------------------------------------------
# Blow out filter
# -----------------------------------------------------------------
def filter_blowout_profiles(H_II_profiles, T_profiles, global_parameters,
                            He_II_profiles=None, He_III_profiles=None, threshold=0.9):

    """
    This function filters out all blow-out profiles, i.e. removes all hydrogen ionisation profiles
    whose ionisation front is beyond the edge of the computing grid. The same action is
    performed for their corresponding temperature and helium counterparts.

    Args:
        H_II_profiles: numpy objects containing the hydrogen profiles
        T_profiles:  numpy objects containing the Temperature profiles
        global_parameters: numpy objects containing the RT parameters
        He_II_profiles: numpy objects containing the helium II profiles
        He_III_profiles: numpy objects containing the helium III profiles
        threshold:  Filter threshold [0,1], should be close to 1.

    Returns:
        Filtered numpy objects

    """

    # A blow out is a x_{H_{II}} profile for which all array values are above a certain threshold (e.g. < 0.9)

    # 0) for each profile find the min value
    profile_minima = np.min(H_II_profiles, axis=1)

    # 1) find all profiles for which the min > threshold
    deletion_indices = np.where(profile_minima > threshold)[0]

    H_II_profiles = np.delete(H_II_profiles, deletion_indices, axis=0)
    T_profiles = np.delete(T_profiles, deletion_indices, axis=0)
    if He_II_profiles is not None:
        He_II_profiles = np.delete(He_II_profiles, deletion_indices, axis=0)
    if He_III_profiles is not None:
        He_III_profiles = np.delete(He_III_profiles, deletion_indices, axis=0)
    global_parameters = np.delete(global_parameters, deletion_indices, axis=0)

    print("\nBlow-out filter: Deleting a total of {} samples. {} remaining.".format(len(deletion_indices),
                                                                                    len(H_II_profiles)))

    if He_II_profiles is None or He_III_profiles is None:
        return H_II_profiles, T_profiles, global_parameters
    else:
        return H_II_profiles, T_profiles, He_II_profiles, He_III_profiles, global_parameters
