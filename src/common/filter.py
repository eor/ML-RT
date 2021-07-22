import numpy as np


# -----------------------------------------------------------------
# file for all filter functions
# -----------------------------------------------------------------
def filter_blowout_profiles(H_profiles, T_profiles, global_parameters, threshold=0.9):
    """
    This function filters out all blow-out profiles, i.e. removes all hydrogen ionisation profiles
    whose ionisation front is beyond the edge of the computing grid. The same action is
    performed for their corresponding temperature counterparts.

    Args:
        numpy objects containing the hydrogen, temperature profiles, and parameters
        threshold : Filter threshold [0,1], should be close to 1.
    """

    # identify all blow-outs
    # blow out is a x_{H_{II}} profile for which all array values are above a certain threshold (e.g. < 0.9)

    # 0) for each profile find the min value
    profile_minima = np.min(H_profiles, axis=1)

    # 1) find all profiles for which the min > threshold
    deletion_indices = np.where(profile_minima > threshold)[0]

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

