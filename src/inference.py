import argparse
import torch
import torch.nn as nn
import numpy as np

import os.path as osp

from models.mlp import *
from common.utils import *
import common.parameter_settings as ps
from torch.autograd import Variable


# -----------------------------------------------------------------
# hard-coded parameters (for now)
# -----------------------------------------------------------------
DATA_PRODUCTS_DIR = 'data_products'
SCALE_PARAMETERS = True

# -----------------------------------------------------------------
#  global  variables :-|
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
#  GAN
# -----------------------------------------------------------------
def inference_cgan(run_dir, model_file_name, parameters):
    """
    Function to use user specified parameters with the CGAN
    trained generator in inference mode.
    
    Returns and array of generated profiles    
    """

    print('Running inference for the CGAN generator')

    # TODO: implement this function once we save the generator models


# -----------------------------------------------------------------
#  CVAE
# -----------------------------------------------------------------
def inference_cvae(run_dir, model_file_name, parameters):
    """
    Function to use user specified parameters with the CVAE in inference mode.

    Returns and array of generated profiles
    """
    print('Running inference for the CVAE (decoder)')

    # TODO: implement this function once we save the CVAE models


# -----------------------------------------------------------------
#  MLP
# -----------------------------------------------------------------
def inference_mlp(run_dir, model_file_name, parameters):
    """
    Function to use user specified parameters with the MLP in inference mode.

    Returns and array of generated profiles
    """

    print('Running inference for the MLP')

    config = utils_load_config(run_dir)

    # set up parameters
    if config.n_parameters == 5:
        parameter_limits = ps.p5_limits

    if config.n_parameters == 8:
        parameter_limits = ps.p8_limits

    if SCALE_PARAMETERS:
        parameters = utils_scale_parameters(limits=parameter_limits, parameters=parameters)

    # convert
    parameters = torch.from_numpy(parameters)
    parameters = Variable(parameters.type(FloatTensor))

    # prepare model
    if config.model == 'MLP1':
        model = MLP1(config)
    elif config.model == 'MLP2':
        model = MLP2(config)
    else:
        print('Error. Check if you are using the right model. Exiting.')
        exit(1)

    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # run inference
    with torch.no_grad():

        output = model(parameters)

    return output


# -----------------------------------------------------------------
#  functions for test runs
# -----------------------------------------------------------------
def inference_test_run_mlp():
    """
    Example function to demonstrate how to use the fully trained MLP with custom parameters.

    Returns nothing so far
    """
    # MLP test
    run_dir = './output_mlp/run_2021_07_22__22_51_22/'
    model_file_name = 'best_model_T_8_epochs.pth.tar'

    p = np.zeros((1, 5))  # has to be 2D array because of BatchNorm

    p[0][0] = 12.0  # M_halo
    p[0][1] = 9.0  # redshift
    p[0][2] = 10.0  # source Age
    p[0][3] = 1.0  # qsoAlpha
    p[0][4] = 0.2  # starsEscFrac

    profile = inference_mlp(run_dir, model_file_name, p)

    # save, plot etc

    # TODO: utils_save_single_profile(profile, path, file_name)
    # plots should be done elsewhere by hand  


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    inference_test_run_mlp()
