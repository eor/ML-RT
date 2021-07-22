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
def inference_gan():

    print('Hello there')


# -----------------------------------------------------------------
#  MLP
# -----------------------------------------------------------------
def inference_mlp(run_dir, model_file_name, input_data):

    print('Hello there')

    config = utils_load_config(run_dir)

    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)

    if config.n_parameters == 5:
        parameter_limits = ps.p5_limits

    if config.n_parameters == 8:
        parameter_limits = ps.p8_limits

    # TODO: don't forget to normalise parameters


    if config.model == 'MLP1':
        model = MLP1(config)
    elif config.model == 'MLP2':
        model = MLP2(config)
    else:
        print('Error. Check if you are using the right model. Exiting.')
        exit(1)

    model.load_state_dict(torch.load(model_path))

    model.eval()
    x = Variable(torch.from_numpy(input_data))

    x = x.double()

    test_parameters = Variable(x.type(FloatTensor))


    # model = model   # import pdb; pdb.set_trace()


    with torch.no_grad():

        output = model(test_parameters)

    # print(type(x))
    # print(type(input_data))




    # for each input parameter vector, get results, save as *.npy files


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # MLP test
    run_dir = './output_mlp/run_2021_07_22__22_51_22/'
    input_data = np.random.random(5)

    model_file_name = 'best_model_T_8_epochs.pth.tar'

    inference_mlp(run_dir, model_file_name, input_data)



