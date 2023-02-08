"""
 About this file: This is a one-off script to generate some specific inference
 data for hand-selected input values for the first paper.
"""
import torch
# import torch.nn as nn
import numpy as np
import os.path as osp

from models.mlp import *

from common.utils import *
from common.settings import *
from common.clock import Clock
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
    FloatTensor = torch.cuda.FloatTensor
else:
    cuda = False
    device = torch.device("cpu")
    FloatTensor = torch.FloatTensor


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
    config = utils_load_config(path=pretrained_models_dir, file_name=config_file_name)

    # set up parameters
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
        raise Exception('Error. Check if you are using the right model. Exiting.')

    model.load_state_dict(torch.load(f=model_path, map_location=device))
    model.to(device)
    model.eval()

    output = {}
    output_time = None

    # run inference
    with torch.no_grad():
        gen_profile = model(parameters)
        output[profile_type] = gen_profile

    if measure_time:
        if cuda:
            output_time = {}
            clock = Clock()
            avg_time, std_time = clock.get_time(model, [parameters])
            output_time['avg_time'] = avg_time
            output_time['std_time'] = std_time
        else:
            print('\nTime can only be measured when using a GPU. Skipping. \n')

    return output, output_time


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    print('Let\'s run inference again!')

    paper_data_directory = '../paper_data/'
    models_dir = osp.join(paper_data_directory, PRETRAINED_MODELS_DIR)

    input_parameters = np.zeros((3, 8))

    input_parameters[0] = [10.0, 9.0, 10.0, 1.0, 0.1, 0.1, 2.35, 2.0]      # example 1
    input_parameters[1] = [13.0, 6.0, 10.0, 1.0, 0.1, 0.1, 2.35, 2.0]      # example 2
    input_parameters[2] = [8.0, 12.5, 10.0, 1.0, 0.1, 0.1, 2.35, 2.0]      # example 3

    results_123_H, _ = inference_mlp(parameters=input_parameters.copy(),
                                     profile_type="H",
                                     pretrained_models_dir=models_dir)

    results_123_T, _ = inference_mlp(parameters=input_parameters.copy(),
                                     profile_type="T",
                                     pretrained_models_dir=models_dir)

    input_parameters = np.zeros((5, 8))
    input_parameters[0] = [10.0, 7.0,  4.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    input_parameters[1] = [10.0, 7.0,  8.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    input_parameters[2] = [10.0, 7.0, 12.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    input_parameters[3] = [10.0, 7.0, 16.0, 1.0, 0.1, 0.0, 2.35, 2.0]
    input_parameters[4] = [10.0, 7.0, 19.9, 1.0, 0.1, 0.0, 2.35, 2.0]

    results_4_H, _ = inference_mlp(parameters=input_parameters.copy(),
                                   profile_type="H",
                                   pretrained_models_dir=models_dir)

    results_4_T, _ = inference_mlp(parameters=input_parameters.copy(),
                                   profile_type="T",
                                   pretrained_models_dir=models_dir
                                   )

    results_123_H = results_123_H['H'].cpu().numpy()
    results_123_T = results_123_T['T'].cpu().numpy()

    results_4_H = results_4_H['H'].cpu().numpy()
    results_4_T = results_4_T['T'].cpu().numpy()

    np.save(file="inference_123_H.npy", arr=results_123_H)
    np.save(file="inference_123_T.npy", arr=results_123_T)

    np.save(file="inference_4_H.npy", arr=results_4_H)
    np.save(file="inference_4_T.npy", arr=results_4_T)
