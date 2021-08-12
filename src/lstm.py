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

from models.lstm import *
from common.dataset import RTdata
from common.filter import *
from common.utils import *
from common.analysis import *
import common.parameter_settings as ps
from common.utils import utils_compute_dtw, utils_compute_mse
from common.sdtw_cuda_loss import SoftDTW


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
CUT_PARAMTER_SPACE = True
ENABLE_EARLY_STOPPING = True

DATA_PRODUCTS_DIR = 'data_products'
PLOT_DIR = 'plots'

# -----------------------------------------------------------------
#  global  variables :-|
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
#  loss function
# -----------------------------------------------------------------
soft_dtw_loss = SoftDTW(use_cuda=True, gamma=0.1)


def lstm_loss_function(func, gen_x, real_x, config):
    if func == 'dtw' and cuda:
        # profile tensors are of shape [batch size, profile length]
        # soft dtw wants input of shape [batch size, 1, profile length]
        if len(gen_x.size()) != 3:
            loss = soft_dtw_loss(gen_x.unsqueeze(
                1), real_x.unsqueeze(1)).mean()
        else:
            loss = soft_dtw_loss(gen_x, real_x).mean()
    else:
        loss = F.mse_loss(input=gen_x, target=real_x, reduction='mean')
    return loss


# -----------------------------------------------------------------
#   use lstm with test or val set
# -----------------------------------------------------------------
def lstm_run_evaluation(current_epoch, data_loader, model, path, config, print_results=False, save_results=False, best_model=False):
    """
    function runs the given dataset through the lstm, returns mse_loss and dtw_loss, 
    and saves the results as well as ground truth to file, if in test mode.

    Args:
        epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
        save_reults: whether to save actual and generated profiles locally (default: False)
        best_model: flag for testing on best model
    """

    if save_results:
        mode = 'Test'
        print("\033[94m\033[1mTesting the LSTM now at epoch %d \033[0m" %
              (current_epoch))
    else:
        mode = 'Validation'

    if cuda:
        model.cuda()

    if save_results:
        test_profiles_gen_all = torch.tensor([], device=device)
        test_profiles_true_all = torch.tensor([], device=device)
        test_parameters_true_all = torch.tensor([], device=device)

    # Note: ground truth data could be obtained elsewhere but by getting it from the data loader here
    # we don't have to worry about randomisation of the samples.

    model.eval()

    loss_dtw = 0.0
    loss_mse = 0.0

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):
            batch_size = profiles.shape[0]

            # configure input
            profiles_true = Variable(profiles.type(FloatTensor))
            parameters = Variable(parameters.type(FloatTensor))

            # inference
            profiles_gen = model(parameters)

            # compute loss via soft dtw
            # profile tensors are of shape [batch size, profile length]
            # soft dtw wants input of shape [batch size, 1, profile length]

            dtw = lstm_loss_function(
                'dtw', profiles_true, profiles_gen, config)
            loss_dtw += dtw

            # compute loss via MSE:
            mse = lstm_loss_function(
                'mse', profiles_true, profiles_gen, config)
            loss_mse += mse

            if save_results:
                # collate data
                test_profiles_gen_all = torch.cat(
                    (test_profiles_gen_all, profiles_gen), 0)
                test_profiles_true_all = torch.cat(
                    (test_profiles_true_all, profiles_true), 0)
                test_parameters_true_all = torch.cat(
                    (test_parameters_true_all, parameters), 0)

    # mean of computed losses
    loss_mse = loss_mse / len(data_loader)
    loss_dtw = loss_dtw / len(data_loader)

    if print_results:
        print("%s results: MSE: %e DTW %e" % (mode, loss_mse, loss_dtw))

    if save_results:
        # move data to CPU, re-scale parameters, and write everything to file
        test_profiles_gen_all = test_profiles_gen_all.cpu().numpy()
        test_profiles_true_all = test_profiles_true_all.cpu().numpy()
        test_parameters_true_all = test_parameters_true_all.cpu().numpy()

        test_parameters_true_all = utils_rescale_parameters(
            limits=parameter_limits, parameters=test_parameters_true_all)

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        utils_save_test_data(
            parameters=test_parameters_true_all,
            profiles_true=test_profiles_true_all,
            profiles_gen=test_profiles_gen_all,
            path=path,
            profile_choice=config.profile_type,
            epoch=current_epoch,
            prefix=prefix
        )

    return loss_mse.item(), loss_dtw.item()


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
    global_parameter_file_path = utils_join_path(
        config.data_dir, GLOBAL_PARAMETER_FILE)

    H_profiles = np.load(H_profile_file_path)
    T_profiles = np.load(T_profile_file_path)
    global_parameters = np.load(global_parameter_file_path)

    # -----------------------------------------------------------------
    # OPTIONAL: Filter (blow-out) profiles
    # -----------------------------------------------------------------
    if USE_BLOWOUT_FILTER:
        H_profiles, T_profiles, global_parameters = filter_blowout_profiles(
            H_profiles, T_profiles, global_parameters)

    if CUT_PARAMTER_SPACE:
        H_profiles, T_profiles, global_parameters = filter_cut_parameter_space(
            H_profiles, T_profiles, global_parameters)

    # -----------------------------------------------------------------
    # log space?
    # -----------------------------------------------------------------
    if USE_LOG_PROFILES:
        # add a small number to avoid trouble
        H_profiles = np.log10(H_profiles + 1.0e-6)
        T_profiles = np.log10(T_profiles)

    # -----------------------------------------------------------------
    # shuffle / rescale parameters
    # -----------------------------------------------------------------
    if SCALE_PARAMETERS:
        global_parameters = utils_scale_parameters(
            limits=parameter_limits, parameters=global_parameters)

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
    training_data = RTdata(profiles, global_parameters,
                           split='train', split_frac=SPLIT_FRACTION)
    validation_data = RTdata(profiles, global_parameters,
                             split='val', split_frac=SPLIT_FRACTION)
    testing_data = RTdata(profiles, global_parameters,
                          split='test', split_frac=SPLIT_FRACTION)

    train_loader = DataLoader(
        training_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=config.batch_size)
    test_loader = DataLoader(testing_data, batch_size=config.batch_size)

    # -----------------------------------------------------------------
    # initialise model + check for CUDA
    # -----------------------------------------------------------------
    # if config.model == 'MLP2':
    model = LSTM2(config, device)
    # else:
    #     model = MLP1(config)
    # if True:
    #     return
    if cuda:
        model.cuda()
    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    # return
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr
        # betas=(config.b1, config.b2)
    )

    # -----------------------------------------------------------------
    # book keeping arraysSCALE_PARAMETERS
    # -----------------------------------------------------------------
    train_loss_array = np.empty(0)
    val_loss_mse_array = np.empty(0)
    val_loss_dtw_array = np.empty(0)

    # -----------------------------------------------------------------
    # keep the model with min validation loss
    # -----------------------------------------------------------------
    best_model = copy.deepcopy(model)
    best_loss_mse = np.inf
    best_loss_dtw = np.inf
    best_epoch_mse = 0
    best_epoch_dtw = 0

    # -----------------------------------------------------------------
    # Early Stopping Criteria
    # -----------------------------------------------------------------

    # number of epochs to try to before ending training
    early_stopping_threshold = 10
    n_epoch_without_improvement = 0
    stopped_early = False
    epochs_trained = -1

    # -----------------------------------------------------------------
    # Loss function to use
    # -----------------------------------------------------------------
    train_loss_func = 'mse'  # 'mse' or 'dtw'

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):

        epoch_loss = 0

        # set model mode
        model.train()

        for i, (profiles, parameters) in enumerate(train_loader):

            # in case one batch is smaller (most likely last batch)
            batch_size = profiles.shape[0]
            # configure input
            real_profiles = Variable(profiles.type(FloatTensor))
            real_parameters = Variable(parameters.type(FloatTensor))

            # zero the gradients on each iteration
            optimizer.zero_grad()

            # generate a batch of profiles
            gen_profiles = model(real_parameters)
            loss = lstm_loss_function(
                train_loss_func, gen_profiles, real_profiles, config)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # end-of-epoch book keeping
        train_loss = epoch_loss / len(train_loader)
        train_loss_array = np.append(train_loss_array, train_loss)

        # validation & save the best performing model
        val_loss_mse, val_loss_dtw = lstm_run_evaluation(
            current_epoch=epoch,
            data_loader=val_loader,
            model=model,
            path=data_products_path,
            config=config,
            print_results=False,
            save_results=False,
            best_model=False
        )

        val_loss_mse_array = np.append(val_loss_mse_array, val_loss_mse)
        val_loss_dtw_array = np.append(val_loss_dtw_array, val_loss_dtw)

        if val_loss_mse < best_loss_mse:
            best_loss_mse = val_loss_mse
            best_model = copy.deepcopy(model)
            best_epoch_mse = epoch
            n_epoch_without_improvement = 0
        else:
            n_epoch_without_improvement += 1

        if val_loss_dtw < best_loss_dtw:
            best_loss_dtw = val_loss_dtw
            best_epoch_dtw = epoch

        print(
            "[Epoch %d/%d] [Train loss: %e] [Validation loss MSE: %e] [Validation loss DTW: %e] "
            "[Best_epoch (mse): %d] [Best_epoch (dtw): %d]"
            % (epoch, config.n_epochs, train_loss, val_loss_mse, val_loss_dtw,
               best_epoch_mse, best_loss_dtw)
        )

        if ENABLE_EARLY_STOPPING and n_epoch_without_improvement >= early_stopping_threshold:
            print("\033[96m\033[1m\nStopping Early\033[0m\n")
            stopped_early = True
            epochs_trained = epoch
            break

        # check for testing criterion
        # if epoch % config.testing_interval == 0 or epoch == config.n_epochs:
            # lstm_run_evaluation(epoch, test_loader, model, data_products_path, config, print_results=True, save_results = True, best_model=False)
    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    # -----------------------------------------------------------------
    # Save best model and loss functions
    # -----------------------------------------------------------------
    # TODO: save checkpoint for further training?
    # checkpoint = {
    #     'epoch': config.n_epochs,
    #     'state_dict': best_model.state_dict(),
    #     'bestLoss': best_loss,
    #     'optimizer': optimizer.state_dict(),
    #     }

    # saving the best model using default number of epochs and not the best epoch
    utils_save_model(best_model.state_dict(), data_products_path,
                     config.profile_type, config.n_epochs)
    # utils_save_model(best_model_state, data_products_path, config.profile_type, best_epoch)

    utils_save_loss(train_loss_array, data_products_path,
                    config.profile_type, config.n_epochs, prefix='train')
    if train_loss_func == 'mse':
        utils_save_loss(val_loss_mse_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')
    else:
        utils_save_loss(val_loss_dtw_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')

    # -----------------------------------------------------------------
    # Evaluate the best model by using the test set
    # -----------------------------------------------------------------
    # saving best model with best_model = false for now. As we are not exporting the best_epoch
    best_test_mse, best_test_dtw = lstm_run_evaluation(config.n_epochs, test_loader, best_model, data_products_path,
                                                       config, print_results=True, save_results=True, best_model=False)

    # -----------------------------------------------------------------
    # Save some results to config object for later use
    # -----------------------------------------------------------------
    setattr(config, 'best_epoch', best_epoch_mse)
    setattr(config, 'best_epoch_dtw', best_epoch_dtw)

    setattr(config, 'best_val_mse', best_loss_mse)
    setattr(config, 'best_val_dtw', best_loss_dtw)

    setattr(config, 'best_test_mse', best_test_mse)
    setattr(config, 'best_test_dtw', best_test_dtw)

    setattr(config, 'stopped_early', stopped_early)
    setattr(config, 'epochs_trained', epochs_trained)

    # -----------------------------------------------------------------
    # Overwrite config object
    # -----------------------------------------------------------------
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # TODO: (!optional) save training time

    # finished
    print('\nAll done!')

    # -----------------------------------------------------------------
    # Optional: analysis
    # -----------------------------------------------------------------
    if config.analysis:
        print("\n\033[96m\033[1m\nRunning analysis\033[0m\n")

        analysis_loss_plot(config)
        analysis_auto_plot_profiles(config, k=10, prefix='best')
        analysis_parameter_space_plot(config, prefix='best')


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='ML-RT - Cosmological radiative transfer with neural networks (MLP)')

    # arguments for data handling
    parser.add_argument('--data_dir', type=str,
                        metavar='(string)', help='Path to data directory')

    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument("--testing_interval", type=int,
                        default=100, help="epoch interval between testing runs")

    # physics related arguments
    parser.add_argument('--profile_type', type=str, default='H', metavar='(string)',
                        help='Select H for neutral hydrogen fraction or T for temperature profiles (default: H)')

    parser.add_argument("--profile_len", type=int, default=1500,
                        help="number of profile grid points")

    parser.add_argument("--n_parameters", type=int, default=8,
                        help="number of RT parameters (5 or 8)")

    # network model switch
    parser.add_argument('--model', type=str, default='mlp1', metavar='(string)',
                        help='Pick a model: MLP1 (default) or MLP2')

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches (default=32)")

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

    parser.add_argument("--dropout_value", type=float,
                        default=0.25, help="dropout probability, default=0.25 ")

    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate, default=0.0002 ")

    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    # momentum?

    # etc
    parser.add_argument("--analysis", dest='analysis',
                        action='store_true', help="automatically generate some plots")
    parser.add_argument("--no-analysis", dest='analysis',
                        action='store_false', help="do not run analysis (default)")
    parser.set_defaults(analysis=False)

    my_config = parser.parse_args()

    # sanity checks
    if my_config.data_dir is None:
        print('\nError: Parameter data_dir must not be empty. Exiting.\n')
        argparse.ArgumentParser().print_help()
        exit(1)

    if my_config.n_parameters not in [5, 8]:
        print(
            '\nError: Number of parameters can currently only be either 5 or 8. Exiting.\n')
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
