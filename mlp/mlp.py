import argparse
import os
# import numpy as np
# import math


from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

from models import *
from dataset import RTdata
from utils import *
import parameter_settings as ps


# -----------------------------------------------------------------
# hard-coded parameters (for now)
# -----------------------------------------------------------------
H_PROFILE_FILE = 'data_Hprofiles.npy'
T_PROFILE_FILE = 'data_Tprofiles.npy'
GLOBAL_PARAMETER_FILE = 'data_parameters.npy'

SPLIT_FRACTION = (0.90, 0.09, 0.01) # train, val, test.
SHUFFLE = True
SHUFFLE_SEED = 42

SCALE_PARAMETERS = True
USE_LOG_PROFILES = True
USE_BLOWOUT_FILTER = True

DATA_PRODUCTS_DIR = 'data_products'
PLOT_DIR = 'plots'

# -----------------------------------------------------------------
#  global  variables :-|
# -----------------------------------------------------------------
parameter_limits = list()


# -----------------------------------------------------------------
#  loss function
# -----------------------------------------------------------------
def mlp_loss_function(gen_x, real_x):
    mse = F.mse_loss(input=gen_x, target=real_x.view(-1, 1500), reduction='mean')

    return mse


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
#   use mlp with test set
# -----------------------------------------------------------------
def mlp_run_testing(epoch, data_loader, model, path, config):
    """
    function runs the test data set through the mlp, saves results as well as ground truth to file

    Args:
        epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
    """

    print("\033[94m\033[1mTesting the MLP now at epoch %d \033[0m"%(epoch+1))

    if cuda:
        model.cuda()

    test_profiles_gen_all = torch.tensor([], device=device)
    test_profiles_true_all = torch.tensor([], device=device)
    test_parameters_true_all = torch.tensor([], device=device)

    # Note: ground truth data could be obtained elsewhere but by getting it from the data loader here
    # we don't have to worry about randomisation of the samples.

    model.eval()

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):
            batch_size = profiles.shape[0]

            # configure input
            test_profiles_true = Variable(profiles.type(FloatTensor))
            test_parameters = Variable(parameters.type(FloatTensor))

            # inference
            test_profiles_gen = model(test_parameters)

            # collate data
            test_profiles_gen_all = torch.cat((test_profiles_gen_all, test_profiles_gen), 0)
            test_profiles_true_all = torch.cat((test_profiles_true_all, test_profiles_true), 0)
            test_parameters_true_all = torch.cat((test_parameters_true_all, test_parameters), 0)

    # move data to CPU, re-scale parameters, and write everything to file
    test_profiles_gen_all = test_profiles_gen_all.cpu().numpy()
    test_profiles_true_all = test_profiles_true_all.cpu().numpy()
    test_parameters_true_all = test_parameters_true_all.cpu().numpy()

    test_parameters_true_all = utils_rescale_parameters(limits=parameter_limits, parameters=test_parameters_true_all)

    utils_save_test_data(
        parameters=test_parameters_true_all,
        profiles_true=test_profiles_true_all,
        profiles_gen=test_profiles_gen_all,
        path=path,
        profile_choice=config.profile_type,
        epoch=epoch,
        prefix='test'
    )


# -----------------------------------------------------------------
#   run validation
# -----------------------------------------------------------------
def mlp_run_validation(epoch, data_loader, model, config):
    """
    function runs validation data set to prevent overfitting of the model.
    Returns averaged validation loss.

    Args:
        epoch: current epoch
        data_loader: data loader used for the inference, here, validation set
        model: current model state
        config: config object with user supplied parameters
    """

    if cuda:
        model.cuda()

    val_loss = 0.0

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):
            batch_size = profiles.shape[0]

            # configure input
            val_profiles_true = Variable(profiles.type(FloatTensor))
            val_parameters = Variable(parameters.type(FloatTensor))

            # inference
            val_profiles_gen = model(val_parameters)

            loss = mlp_loss_function(val_profiles_gen, val_profiles_true)

            val_loss += loss.item()

    return val_loss/len(data_loader)


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):

    # -----------------------------------------------------------------
    # create unique output path and run directories
    # -----------------------------------------------------------------
    run_id = 'run_' + utils_get_current_timestamp()
    config.out_dir = os.path.join(config.out_dir, run_id)

    utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_write_config_to_file(config)

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
        H_profiles, T_profiles, global_parameters = utils_filter_profiles(H_profiles, T_profiles, global_parameters)

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
        #adversarial_loss.cuda()

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

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(config.n_epochs):

        epoch_loss = 0

        # set model mode
        model.train()

        for i, (profiles, parameters) in enumerate(train_loader):

            batch_size = profiles.shape[0]      # in case one batch is smaller (most likely last batch)

            # configure input
            real_profiles = Variable(profiles.type(FloatTensor))
            real_parameters = Variable(parameters.type(FloatTensor))

            # zero the gradients on each iteration
            optimizer.zero_grad()

            # generate a batch of profiles
            gen_profiles = model(real_parameters)

            loss = mlp_loss_function(gen_profiles, real_profiles)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # end-of-epoch book keeping
        train_loss = epoch_loss / len(train_loader.dataset)
        train_loss_array = np.append(train_loss_array, train_loss)

        # validation & save the best performing model
        val_loss = mlp_run_validation(epoch, val_loader, model, config)
        val_loss_array = np.append(val_loss_array, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        print(
            "[Epoch %d/%d] [Train loss: %e] [Validation loss: %e] [Best epoch: %d]"
            % (epoch+1, config.n_epochs,  train_loss, val_loss, best_epoch)
        )

        # check for testing criterion
        if (epoch+1) % config.testing_interval == 0 or epoch+1 == config.n_epochs:

            mlp_run_testing(epoch, test_loader, model, data_products_path, config)

    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    # -----------------------------------------------------------------
    # Save best model and loss functions
    # -----------------------------------------------------------------
    best_model_state = {
        'epoch': config.n_epochs,
        'state_dict': best_model.state_dict(),
        'bestLoss': best_loss,
        'optimizer': optimizer.state_dict(),
        }

    utils_save_model(best_model_state, data_products_path, config.profile_type, best_epoch)

    utils_save_loss(train_loss_array, data_products_path, config.profile_type, config.n_epochs, prefix='train')
    utils_save_loss(val_loss_array, data_products_path, config.profile_type, config.n_epochs, prefix='val')

    # finished
    print('\nAll done!')


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()

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

    if my_config.n_parameters == 8:
        parameter_limits = ps.p8_limits

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
