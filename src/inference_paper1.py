import torch
import torch.nn as nn
import numpy as np
import os.path as osp

from models.mlp import *


from common.utils import *
from common.settings import *
from common.clock import Clock
from common.plot import plot_inference_profiles, plot_inference_time_evolution
import common.settings_parameters as sp

from torch.autograd import Variable

# -----------------------------------------------------------------
#  global  variables
# -----------------------------------------------------------------
parameter_limits = list()

# -----------------------------------------------------------------
#  CUDA available?
# -----------------------------------------------------------------
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
else:
    cuda = False
    device = torch.device("cpu")

# -----------------------------------------------------------------
#  global FloatTensor instance
# -----------------------------------------------------------------
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# -----------------------------------------------------------------
#  MLP
# -----------------------------------------------------------------
def inference_mlp(parameters, profile_type, pretrained_models_dir, model_file_name=None,
                  config_file_name=None, measure_time=False):
    """
    Function to use user specified parameters with the MLP in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the MLP')

    if model_file_name is None:
        model_file_name = F'best_model_{profile_type}_MLP.pth.tar'
    if config_file_name is None:
        config_file_name = F'config_{profile_type}_MLP.dict'

    model_path = osp.join(pretrained_models_dir, model_file_name)
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)

    # set up parameters
    if config.n_parameters == 5:
        parameter_limits = sp.p5_limits

    if config.n_parameters == 8:
        parameter_limits = sp.p8_limits

    if SCALE_PARAMETERS:
        parameters = utils_scale_parameters(limits=parameter_limits, parameters=parameters)

    # convert numpy arrays to tensors
    parameters = torch.from_numpy(parameters).to(device)
    parameters = Variable(parameters.type(FloatTensor))

    # prepare model
    if config.model == 'MLP1':
        model = MLP1(config)
    elif config.model == 'MLP2':
        model = MLP2(config)
    else:
        print('Error. Check if you are using the right model. Exiting.')
        exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    output = {}
    output_time = None

    # run inference
    with torch.no_grad():
        gen_profile = model(parameters)
        output[profile_type] = gen_profile

    output_time = None
    if measure_time:
        if cuda:
            output_time = {}
            clock = Clock()
            avg_time, std_time = clock.get_time(model, [parameters])
            output_time['avg_time'] = avg_time
            output_time['std_time'] = std_time
        else:
            print('\nTime can only be measured when on GPU. skipping. \n')

    return output, output_time


# -----------------------------------------------------------------
#  functions for test runs - MLP
# -----------------------------------------------------------------
def inference_test_run_mlp():
    """
    Example function to demonstrate how to use the fully trained MLP with custom parameters.

    Returns nothing so far
    """
    # MLP test
    run_dir = './test/MLP_H_run/'
    model_file_name = 'best_model_H_1105_epochs.pth.tar'

    p = np.zeros((1, 5))  # has to be 2D array because of BatchNorm

    p[0][0] = 12.0  # M_halo
    p[0][1] = 9.0  # redshift
    p[0][2] = 10.0  # source Age
    p[0][3] = 1.0  # qsoAlpha
    p[0][4] = 0.2  # starsEscFrac

    profile_MLP, output_time = inference_mlp(run_dir, model_file_name, p, measure_time=True)

    if output_time is not None:
        print('Inference time for MLP1: %e ± %e ms' %(output_time['avg_time'], output_time['std_time']))


def inference_time_evolution(paper_data_directory,
                             pretrained_models_dir=None,
                             models_to_use=['MLP', 'CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'],
                             measure_time=False):

    """
    Function to load the sde data and run it through the
    inference_model_comparison function for H and T profiles.
    """

    base_path = osp.join(paper_data_directory,
                         ARCH_COMPARISON_DIR, SD_RUNS_DIR)

    inference_plots_path = osp.join(
        paper_data_directory, ARCH_COMPARISON_DIR, INFERENCE_DIR)
    print(inference_plots_path)

    # Create inference plots dir if doesn't exist
    utils_create_output_dirs([inference_plots_path])

    if pretrained_models_dir is None:
        pretrained_models_dir = osp.join(
            paper_data_directory, PRETRAINED_MODELS_DIR)

    # +1 because we will also be plotting simulation along with generated profiles
    concat_profiles_gen_H = np.empty([0, len(models_to_use) + 1, 1500])
    concat_profiles_gen_T = np.empty([0, len(models_to_use) + 1, 1500])
    concat_parameters = np.empty([0, 1, 8])

    for i in range(8, 24, 4):
        # path of actual profiles
        parameter_file_path = osp.join(base_path, 'run_4', F'run_4_t{i}_parameters.npy')
        H_profile_path = osp.join(base_path, 'run_4', F'run_4_t{i}_profile_HII.npy')
        T_profile_path = osp.join(base_path, 'run_4', F'run_4_t{i}_profile_T.npy')
        He_II_profile_path = osp.join(base_path, 'run_4', F'run_4_t{i}_profile_HeII.npy')
        He_III_profile_path = osp.join(base_path, 'run_4', F'run_4_t{i}_profile_HeIII.npy')

        # load the actual profiles
        print('loading profiles for time t=%d\n' % (i))
        parameters = np.load(parameter_file_path)
        H_II_profiles = np.load(H_profile_path)
        T_profiles = np.load(T_profile_path)
        He_II_profiles = np.load(He_II_profile_path)
        He_III_profiles = np.load(He_III_profile_path)
        print('loaded profiles for time t=%d\n' % (i))

        if USE_LOG_PROFILES:
            # add a small number to avoid trouble
            H_II_profiles = np.log10(H_II_profiles + 1.0e-6)
            He_II_profiles = np.log10(He_II_profiles + 1.0e-6)
            He_III_profiles = np.log10(He_III_profiles + 1.0e-6)
            T_profiles = np.log10(T_profiles)

        # return profiles of (batch_size, profile_length). Because batch_size is 1 for us, we will be using
        # this dimension to stack profiles at different time_step
        profiles_gen_H = inference_model_comparison(pretrained_models_dir, 'H', parameters,
                                                    actual_profiles=H_II_profiles,
                                                    models_to_use=models_to_use,
                                                    plot=False,
                                                    plot_output_dir=inference_plots_path,
                                                    prefix='run_%d' % i,
                                                    measure_time=measure_time)

        profiles_gen_T = inference_model_comparison(pretrained_models_dir, 'T', parameters,
                                                    actual_profiles=T_profiles,
                                                    models_to_use=models_to_use,
                                                    plot=False,
                                                    plot_output_dir=inference_plots_path,
                                                    prefix='run_%d' % i,
                                                    measure_time=measure_time)

        concat_profiles_gen_H = np.concatenate((concat_profiles_gen_H, profiles_gen_H), axis=0)
        concat_profiles_gen_T = np.concatenate((concat_profiles_gen_T, profiles_gen_T), axis=0)
        concat_parameters = np.concatenate((concat_parameters, parameters[np.newaxis, np.newaxis, :]), axis=0)


    # TODO: write data if desired, to plot elsewhere

    plot_inference_time_evolution(concat_profiles_gen_H, 'H', concat_parameters, output_dir='./',
                                  labels=['Simulation'] + models_to_use,
                                  file_type='pdf', prefix='run_4')

    plot_inference_time_evolution(concat_profiles_gen_T, 'T', concat_parameters, output_dir='./',
                                  labels=['Simulation'] + models_to_use,
                                  file_type='pdf', prefix='run_4')


# def inference_estimate_number_density_ranges(pretrained_models_dir,
#                                              actual_parameters,
#                                              radius=[0, 250, 750, 1250, 1500],
#                                              measure_time=False):
#
#     constant_n_H_0 = 1.9e-7  # cm-3
#     constant_n_He_0 = 1.5e-8 # cm-3
#
#     if pretrained_models_dir is None:
#         pretrained_models_dir = osp.join(paper_data_directory, PRETRAINED_MODELS_DIR)
#
#     # convert input to 2D form
#     p_2D = actual_parameters.copy()
#     if len(np.shape(p_2D)) != 2:
#         p_2D = p_2D[np.newaxis, :]
#
#     # run inference on the parameters
#     output_profiles, output_time = inference_cmlp(p_2D.copy(), 'C',
#                                                   pretrained_models_dir,
#                                                   measure_time=measure_time)
#
#     # obtain inference profiles for all our parameters
#     # and convert them to numpy arrays.
#     x_H_II = output_profiles['H'].numpy()
#     x_T = output_profiles['T'].numpy()
#     x_He_II = output_profiles['He_II'].numpy()
#     x_He_III = output_profiles['He_III'].numpy()
#
#     # convert log profiles to normal scale
#     x_H_II = np.power(10, x_H_II)
#     x_T = np.power(10, x_T)
#     x_He_II = np.power(10, x_He_II)
#     x_He_III = np.power(10, x_He_III)
#
#     # obtain x_H_I and x_He_I (neutral hydrogen and helium)
#     # from ionisation fractions
#     x_H_I = 1 - x_H_II
#     x_He_I = 1 - x_He_II - x_He_III
#
#     # select redshift from input parameters and recompute the shape.
#     redshift = p_2D[:, 1].reshape((p_2D.shape[0], -1))
#     # compute n_H and n_He for the redshifts
#     n_H = np.power((1 + redshift), 3) * constant_n_H_0
#     n_He = np.power((1 + redshift), 3) * constant_n_He_0
#
#     # compute neutral hydrogen and helium number densities
#     # from ionisation fractions for all radius r.
#     n_H_I = n_H * x_H_I
#     n_He_I = n_He * x_He_I
#     for r in radius:
#         print('\nFor radius (r) = %d'%(r))
#         print("| {:<8} | {:<15} | {:<15} |".format('redshift','avg. n_H_I','avg. n_He_I'))
#         for i in range(len(p_2D)):
#             avg_number_density_hydrogen = np.average(n_H_I[i, :r])
#             avg_number_density_helium = np.average(n_He_I[i, :r])
#             print("| {:<8} | {:<15e} | {:<15e} |".format(redshift[i, 0],
#                                                          avg_number_density_hydrogen,
#                                                          avg_number_density_helium))


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    print('Let\'s run inference!')

    paper_data_directory = '../paper_data/'
    models_to_use = ['MLP', 'CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM']
    pretrained_models_dir = osp.join(paper_data_directory, PRETRAINED_MODELS_DIR)

#     inference_main(paper_data_directory,
#                    pretrained_models_dir=pretrained_models_dir,
#                    models_to_use=models_to_use,
#                    measure_time=False)
#     inference_test_run_cmlp()
    # inference_time_evolution(paper_data_directory,
    #                          pretrained_models_dir=pretrained_models_dir,
    #                          models_to_use=models_to_use,
    #                          measure_time=False)

    # To have a custom run without knowing actual profile
    p = np.zeros((8))  # has to be 2D array because of BatchNorm
    p[0] = 8.825165  # M_halo
    p[1] = 8.285341  # redshift
    p[2] = 14.526998  # source Age
    p[3] = 1.491899   # qsoAlpha
    p[4] = 0.79072833   # qsoEfficiency
    p[5] = 0.48244837  # starsEscFrac
    p[6] = 1.5012491  # starsIMFSlope
    p[7] = 1.5323509  # starsIMFMassMinLog

    # p = np.zeros((5, 8))  # has to be 2D array because of BatchNorm
    # p[0] = [8.825165, 6.0, 14.526998, 1.491899, 0.79072833, 0.48244837, 1.5012491, 1.5323509]
    # p[1] = [8.825165, 8.0, 14.526998, 1.491899, 0.79072833, 0.48244837, 1.5012491, 1.5323509]
    # p[2] = [8.825165, 9.5, 14.526998, 1.491899, 0.79072833, 0.48244837, 1.5012491, 1.5323509]
    # p[3] = [8.825165, 11.0, 14.526998, 1.491899, 0.79072833, 0.48244837, 1.5012491, 1.5323509]
    # p[4] = [8.825165, 13.0, 14.526998, 1.491899, 0.79072833, 0.48244837, 1.5012491, 1.5323509]

    p = np.zeros((3, 8))  # has to be 2D array because of BatchNorm

    p[0] = [10.0, 9.0, 10.0, 1.0, 0.1, 0.1, 2.35, 2.0]      # example 1
    p[0] = [13.0, 6.0, 10.0, 1.0, 0.1, 0.1, 2.35, 2.0]      # example 2
    p[0] = [8.0, 12.5, 10.0, 1.0, 0.1, 0.1, 2.35, 2.0]      # example 3

    results_123_H, _ = inference_mlp(parameters=p, profile_type="H", pretrained_models_dir=pretrained_models_dir)
    results_123_T, _ = inference_mlp(parameters=p, profile_type="T", pretrained_models_dir=pretrained_models_dir)

    p = np.zeros((5, 8))  # has to be 2D array because of BatchNorm
    p[0] = [10.0, 7.0,  4.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    p[0] = [10.0, 7.0,  8.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    p[0] = [10.0, 7.0, 12.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    p[0] = [10.0, 7.0, 16.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    p[0] = [10.0, 7.0, 19.9, 1.0, 0.1, 0.0, 2.35, 2.0]

    results_4_H, _ = inference_mlp(parameters=p, profile_type="H", pretrained_models_dir=pretrained_models_dir)
    results_4_T, _ = inference_mlp(parameters=p, profile_type="T", pretrained_models_dir=pretrained_models_dir)

    results_123_H = results_123_H.numpy()
    results_123_T = results_123_T.numpy()

    results_4_H = results_4_H.numpy()
    results_4_T = results_4_T.numpy()


    import pdb; pdb.set_trace()

    np.save(file="results_123_H.npy", arr=results_123_H)
    np.save(file="results_123_T.npy", arr=results_123_T)

    np.save(file="results_4_H.npy", arr=results_4_H)
    np.save(file="results_4_T.npy", arr=results_4_T)


