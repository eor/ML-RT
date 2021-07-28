import argparse

# import os
# import numpy as np
# import math
# from torchvision import datasets
# import torch.nn as nn
# import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import copy

from models.mlp import *
from common.dataset import RTdata
from common.filter import *
from common.utils import *
from common.analysis import *
import common.parameter_settings as ps


# -----------------------------------------------------------------
# hard-coded parameters (for now)
# -----------------------------------------------------------------
H_PROFILE_FILE = 'data_Hprofiles.npy'
T_PROFILE_FILE = 'data_Tprofiles.npy'
GLOBAL_PARAMETER_FILE = 'data_parameters.npy'

SPLIT_FRACTION = (0.90, 0.09, 0.01)  # train, val, test.
SHUFFLE = True
SHUFFLE_SEED = 42

SCALE_PARAMETERS = True
USE_LOG_PROFILES = True
USE_BLOWOUT_FILTER = True

DATA_PRODUCTS_DIR = 'data_products'
PLOT_DIR = 'plots'

# -----------------------------------------------------------------
#  global  variables
# -----------------------------------------------------------------
parameter_limits = list()
parameter_names_latex = list()


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
#  Loss function
# -----------------------------------------------------------------
def cvae_loss_function(gen_x, real_x, mu, log_var, config):

    """
    Loss function for the CVAE contains two components: 1) MSE to quantify the difference between
    input and output, adn 2) the KLD to force the autoencoder to be more effective, see also
    Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114

    Args:
        gen_x: generated or inferred profile
        real_x: ground truth profile
        mu:   mean
        log_var: log of variance
        config: config object with user supplied parameters

    Returns:
        sum of the two components
    """

    MSE0 = F.mse_loss(input=gen_x, target=real_x.view(-1, config.profile_len), reduction='mean')

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return MSE0 + KLD


# -----------------------------------------------------------------
#  Training
# -----------------------------------------------------------------
# def cvae_train(model, optimizer, trainLoader, device):
#
#     model.train()
#     trainLoss = 0
#
#     for batch_idx, (profiles, parameters) in enumerate(trainLoader):
#
#         profiles = profiles.to(device)
#         parameters = parameters.to(device)
#         optimizer.zero_grad()
#
#         recoveredData, mu, logVar = model(profiles, parameters)
#
#         loss = cvae_loss_function(recoveredData, profiles, mu, logVar, device)
#         loss.backward()
#
#         trainLoss += loss.item()
#         optimizer.step()
#
#         # debug logging
#         #if batch_idx % log_interval == 0:
#         #    print('Train Epoch: {} [{}/{} ({:.0f}%)]   \tLoss: {:.6f}'.format(
#         #        epoch, batch_idx * len(profiles), len(trainLoader.dataset),
#         #        100. * batch_idx / len(trainLoader),
#         #        loss.item() / len(profiles)))
#
#     averageLoss = trainLoss / len(trainLoader.dataset)
#
#     return averageLoss  # float

# -----------------------------------------------------------------
#  Validation
# -----------------------------------------------------------------
# def cvae_validate(model, valLoader, device):
#
#     model.eval()
#     valLoss = 0
#
#     with torch.no_grad():
#         for i, (profiles, parameters) in enumerate(valLoader):
#
#             profiles = profiles.to(device)
#             parameters = parameters.to(device)
#
#             recoveredData, mu, logvar = model(profiles, parameters)
#             valLoss += cvae_loss_function(recoveredData, profiles, mu, logvar, device).item()
#
#
#     averageLoss = valLoss / len(valLoader.dataset)
#
#     return averageLoss  # float

# -----------------------------------------------------------------
#  Testing
# -----------------------------------------------------------------
# def cvae_test(model, testLoader, device):
#
#     model.eval()
#     test_loss = 0
#
#     with torch.no_grad():
#         for i, (profiles, parameters) in enumerate(testLoader):
#
#             profiles = profiles.to(device)
#             parameters = parameters.to(device)
#
#             inferredData, mu, logvar = model(profiles, parameters)
#             test_loss += cvae_loss_function(inferredData, profiles, mu, logvar, device).item()
#
#
#     averageLoss = test_loss / len(testLoader.dataset)
#     print('\nAverage test loss:\t{:.4e}\n'.format(averageLoss))
#
#     # export data
#     profilesTrue = profiles.cpu().numpy()
#     profilesInfer = inferredData.cpu().numpy()
#     parametersTmp = parameters.cpu().numpy()
#
#     p = parametersTmp.copy()
#     p = utils_rescale_parameters(p)
#
#
#     return p, profilesTrue, profilesInfer


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):

    # -----------------------------------------------------------------
    # create unique output path and run directories, save config
    # -----------------------------------------------------------------
    run_id = 'run_' + utils_get_current_timestamp()
    config.out_dir = os.path.join(config.out_dir, run_id)

    utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)
    plot_path = os.path.join(config.out_dir, PLOT_DIR)

    # -----------------------------------------------------------------
    # Check if data files exist / read data and shuffle / rescale parameters
    # -----------------------------------------------------------------
    H_profile_file_path = utils_join_path(config.data_dir, H_PROFILE_FILE)
    T_profile_file_path = utils_join_path(config.data_dir, T_PROFILE_FILE)
    global_parameter_file_path = utils_join_path(config.data_dir, GLOBAL_PARAMETER_FILE)

    H_profiles = np.load(H_profile_file_path)
    T_profiles = np.load(T_profile_file_path)
    global_parameters = np.load(global_parameter_file_path)

    # -----------------------------------------------------------------
    # OPTIONAL: Filter (blow-out) profiles
    # -----------------------------------------------------------------
    if USE_BLOWOUT_FILTER:
        H_profiles, T_profiles, global_parameters = filter_blowout_profiles(H_profiles, T_profiles, global_parameters)

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
        global_parameters = utils_scale_parameters(limits=parameter_limits, parameters=global_parameters)

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

    # -----------------------------------------------------------------
    # initialise model + check for CUDA
    # -----------------------------------------------------------------
    if config.model == 'MLP2':
        model = MLP2(config)
    else:
        model = MLP1(config)

    if cuda:
        model.cuda()

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    # -----------------------------------------------------------------
    # book keeping arrays
    # -----------------------------------------------------------------
    train_loss_array = np.empty(0)
    val_loss_array = np.empty(0)

    # -----------------------------------------------------------------
    # keep the model with min validation loss
    # -----------------------------------------------------------------
    best_model = copy.deepcopy(model)
    best_loss = np.inf
    best_epoch = 0




    # TODO: finish this!


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with neural networks')

    # arguments for data handling
    parser.add_argument('--data_dir', type=str, metavar='(string)', help='Path to data directory')
    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument("--testing_interval", type=int, default=100, help="epoch interval between testing runs")

    # physics related arguments
    parser.add_argument('--profile_type', type=str, default='H', metavar='(string)',
                        help='Select H for neutral hydrogen fraction or T for temperature profiles (default: H)')
    parser.add_argument("--profile_len", type=int, default=1500, help="number of profile grid points")
    parser.add_argument("--n_parameters", type=int, default=8, help="number of RT parameters (5 or 8)")

    parser.add_argument("--gen_parameter_mode", type=int, default=1, help="mode for generating fake parameters (0,1,2)")

    # network model switch
    parser.add_argument('--model', type=str, default='mlp1', metavar='(string)',
                        help='Pick a model: MLP1 (default) or MLP2')

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches (default=32)")

    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true',
                        help="use batch normalisation in network (default)")
    parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false',
                        help="use batch normalisation in network")
    parser.set_defaults(batch_norm=True)

    parser.add_argument("--dropout", dest='dropout', action='store_true',
                        help="use dropout regularisation in network (default)")
    parser.add_argument("--no-dropout", dest='dropout', action='store_false',
                        help="do not use dropout regularisation in network")
    parser.set_defaults(dropout=True)

    parser.add_argument("--dropout_value", type=float, default=0.25, help="dropout probability, default=0.25 ")

    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate, default=0.0002 ")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    # momentum?

    # etc
    parser.add_argument("--analysis", dest='analysis', action='store_true', help="automatically generate some plots")
    parser.add_argument("--no-analysis", dest='analysis', action='store_false', help="do not run analysis (default)")
    parser.set_defaults(analysis=False)

    my_config = parser.parse_args()

    # sanity checks
    if my_config.data_dir is None:
        print('\nError: Parameter data_dir must not be empty. Exiting.\n')
        argparse.ArgumentParser().print_help()
        exit(1)

    if my_config.n_parameters not in [5, 8]:
        print('\nError: Number of parameters can currently only be either 5 or 8. Exiting.\n')
        argparse.ArgumentParser().print_help()
        exit(1)

    if my_config.n_parameters == 5:
        parameter_limits = ps.p5_limits
        parameter_names_latex = ps.p5_names_latex

    if my_config.n_parameters == 8:
        parameter_limits = ps.p8_limits
        parameter_names_latex = ps.p8_names_latex

    if my_config.model not in ['MLP1', 'MLP2']:
        my_config.model = 'MLP1'

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.data_dir = os.path.abspath(my_config.data_dir)

    # run main program
    main(my_config)
