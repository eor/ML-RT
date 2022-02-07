import argparse
import signal

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import copy

from models.cmlp import *
from common.dataset import RTdata
from common.filter import *
from common.utils import *
from common.analysis import *
import common.settings_parameters as ps
from common.settings import *
from common.run import *

from common.soft_dtw_cuda import SoftDTW as SoftDTW_CUDA
from common.soft_dtw import SoftDTW as SoftDTW_CPU


# -----------------------------------------------------------------
#  CUDA available?
# -----------------------------------------------------------------
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    soft_dtw_loss = SoftDTW_CUDA(use_cuda=True, gamma=0.1)
    FloatTensor = torch.cuda.FloatTensor

else:
    cuda = False
    device = torch.device("cpu")
    soft_dtw_loss = SoftDTW_CPU(use_cuda=False, gamma=0.1)
    FloatTensor = torch.FloatTensor


# -----------------------------------------------------------------
#  loss function
# -----------------------------------------------------------------
def cmlp_loss_function(func, gen_x, real_x):

    if func == 'DTW':
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
#  signal handling # TODO put it elsewhere
# -----------------------------------------------------------------

def force_stop_signal_handler(sig, frame):
    global FORCE_STOP
    FORCE_STOP = True
    print("\033[96m\033[1m\nTraining will stop after this epoch. Please wait.\033[0m\n")


# -----------------------------------------------------------------
#   use lstm with test or val set
# -----------------------------------------------------------------
def cmlp_run_evaluation(current_epoch, data_loader, model, path, config,
                        print_results=False, save_results=False, best_model=False):
    """
    function runs the given dataset through the lstm, returns mse_loss and dtw_loss,
    and saves the results as well as ground truth to file, if in test mode.

    Args:
        current_epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
        save_results: whether to save actual and generated profiles locally (default: False)
        best_model: flag for testing on best model
    """

    if save_results:
        print("\033[94m\033[1mTesting the CMLP now at epoch %d \033[0m" % current_epoch)

    if cuda:
        model.cuda()

    if save_results:
        profiles_gen_all = torch.tensor([], device=device)
        profiles_true_all = torch.tensor([], device=device)
        parameters_true_all = torch.tensor([], device=device)

    model.eval()

    loss_dtw, loss_mse = 0.0, 0.0
    loss_dtw_H_II, loss_dtw_T, loss_dtw_He_II, loss_dtw_He_III = 0.0, 0.0, 0.0, 0.0
    loss_mse_H_II, loss_mse_T, loss_mse_He_II, loss_mse_He_III = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, (H_II_profiles, T_profiles, He_II_profiles, He_III_profiles, parameters) in enumerate(data_loader):

            # configure input
            real_H_II_profiles = Variable(H_II_profiles.type(FloatTensor))
            real_T_profiles = Variable(T_profiles.type(FloatTensor))
            real_He_II_profiles = Variable(He_II_profiles.type(FloatTensor))
            real_He_III_profiles = Variable(He_III_profiles.type(FloatTensor))
            real_parameters = Variable(parameters.type(FloatTensor))

            # generate a batch of profiles
            gen_H_II_profiles, gen_T_profiles, gen_He_II_profiles, gen_He_III_profiles = model(real_parameters)

            # compute loss via soft dtw
            dtw_loss_H_II = cmlp_loss_function('DTW', gen_H_II_profiles, real_H_II_profiles)
            dtw_loss_T = cmlp_loss_function('DTW', gen_T_profiles, real_T_profiles)
            dtw_loss_He_II = cmlp_loss_function('DTW', gen_He_II_profiles, real_He_II_profiles)
            dtw_loss_He_III = cmlp_loss_function('DTW', gen_He_III_profiles, real_He_III_profiles)

            dtw = dtw_loss_H_II + dtw_loss_T + dtw_loss_He_II + dtw_loss_He_III
            loss_dtw += dtw.item()
            loss_dtw_H_II += dtw_loss_H_II.item()
            loss_dtw_T += dtw_loss_T.item()
            loss_dtw_He_II += dtw_loss_He_II.item()
            loss_dtw_He_III += dtw_loss_He_III.item()

            # compute loss via MSE:
            mse_loss_H_II = cmlp_loss_function('MSE', gen_H_II_profiles, real_H_II_profiles)
            mse_loss_T = cmlp_loss_function('MSE', gen_T_profiles, real_T_profiles)
            mse_loss_He_II = cmlp_loss_function('MSE', gen_He_II_profiles, real_He_II_profiles)
            mse_loss_He_III = cmlp_loss_function('MSE', gen_He_III_profiles, real_He_III_profiles)

            mse = mse_loss_H_II + mse_loss_T + mse_loss_He_II + mse_loss_He_III
            loss_mse += mse.item()
            loss_mse_H_II += mse_loss_H_II.item()
            loss_mse_T += mse_loss_T.item()
            loss_mse_He_II += mse_loss_He_II.item()
            loss_mse_He_III += mse_loss_He_III.item()

            if save_results:
                # shape of profile_gen and profile_true: (num_samples, num_profiles, length_of_profiles)
                profiles_gen = torch.stack((gen_H_II_profiles, gen_T_profiles,
                                            gen_He_II_profiles, gen_He_III_profiles), dim=1)
                profiles_true = torch.stack((real_H_II_profiles, real_T_profiles,
                                             real_He_II_profiles, real_He_III_profiles), dim=1)
                # collate data
                profiles_gen_all = torch.cat((profiles_gen_all, profiles_gen), 0)
                profiles_true_all = torch.cat((profiles_true_all, profiles_true), 0)
                parameters_true_all = torch.cat((parameters_true_all, real_parameters), 0)

    # mean of computed losses
    loss_mse /= (4 * len(data_loader))
    loss_dtw /= (4 * len(data_loader))

    loss_dtw_H_II /= len(data_loader)
    loss_dtw_T /= len(data_loader)
    loss_dtw_He_II /= len(data_loader)
    loss_dtw_He_III /= len(data_loader)
    stacked_dtw_loss = np.stack((loss_dtw_H_II, loss_dtw_T, loss_dtw_He_II, loss_dtw_He_III))

    loss_mse_H_II /= len(data_loader)
    loss_mse_T /= len(data_loader)
    loss_mse_He_II /= len(data_loader)
    loss_mse_He_III /= len(data_loader)
    stacked_mse_loss = np.stack((loss_mse_H_II, loss_mse_T, loss_mse_He_II, loss_mse_He_III))

    if print_results:
        print("Results: AVERAGE MSE: %e DTW %e" % (loss_mse, loss_dtw))
        print("Results: H_II_profiles MSE: %e DTW %e" % (loss_mse_H_II, loss_dtw_H_II))
        print("Results: T_profiles MSE: %e DTW %e" % (loss_mse_T, loss_dtw_T))
        print("Results: He_II_profiles MSE: %e DTW %e" % (loss_mse_He_II, loss_dtw_He_II))
        print("Results: He_III_profiles MSE: %e DTW %e" % (loss_mse_He_III, loss_dtw_He_III))

    if save_results:
        # move data to CPU, re-scale parameters, and write everything to file
        profiles_gen_all = profiles_gen_all.cpu().numpy()
        profiles_true_all = profiles_true_all.cpu().numpy()
        parameters_true_all = parameters_true_all.cpu().numpy()

        parameters_true_all = utils_rescale_parameters(limits=config.parameter_limits,
                                                       parameters=parameters_true_all)

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        # use profile type
        utils_save_test_data(parameters=parameters_true_all,
                             profiles_true=profiles_true_all,
                             profiles_gen=profiles_gen_all,
                             path=path,
                             profile_choice='C',         # 'C' for combined profiles
                             epoch=current_epoch,
                             prefix=prefix
                             )

    return loss_mse, loss_dtw, stacked_mse_loss, stacked_dtw_loss


# -----------------------------------------------------------------
#  Training
# -----------------------------------------------------------------
def cmlp_train(model, optimizer, train_loader, config):
    """
    This function trains the network for one epoch.
    Returns: averaged training loss. No need to return the model as the optimizer modifies it inplace.

    Args:
        model: current model state
        optimizer: optimizer object to perform the back-propagation
        train_loader: data loader used for the inference, most likely the test set
        config: config object with user supplied parameters

    Returns:
          The average loss
    """

    epoch_loss = epoch_loss_H_II = epoch_loss_T = epoch_loss_He_II = epoch_loss_He_III = 0

    # set model mode
    model.train()

    for i, (H_II_profiles, T_profiles, He_II_profiles, He_III_profiles, parameters) in enumerate(train_loader):
        # configure input
        real_H_II_profiles = Variable(H_II_profiles.type(FloatTensor))
        real_T_profiles = Variable(T_profiles.type(FloatTensor))
        real_He_II_profiles = Variable(He_II_profiles.type(FloatTensor))
        real_He_III_profiles = Variable(He_III_profiles.type(FloatTensor))
        real_parameters = Variable(parameters.type(FloatTensor))

        # zero the gradients on each iteration
        optimizer.zero_grad()

        # generate a batch of profiles
        gen_H_II_profiles, gen_T_profiles, gen_He_II_profiles, gen_He_III_profiles = model(real_parameters)

        # compute loss
        loss_H_II = cmlp_loss_function(config.loss_type, gen_H_II_profiles, real_H_II_profiles)
        loss_T = cmlp_loss_function(config.loss_type, gen_T_profiles, real_T_profiles)
        loss_He_II = cmlp_loss_function(config.loss_type, gen_He_II_profiles, real_He_II_profiles)
        loss_He_III = cmlp_loss_function(config.loss_type, gen_He_III_profiles, real_He_III_profiles)

        loss = loss_H_II + loss_T + loss_He_II + loss_He_III
        loss.backward()
        optimizer.step()

        # sum the loss values
        epoch_loss += loss.item()
        epoch_loss_H_II = loss_H_II.item()
        epoch_loss_T = loss_T.item()
        epoch_loss_He_II = loss_He_II.item()
        epoch_loss_He_III = loss_He_III.item()

    # end-of-epoch book keeping
    train_loss = epoch_loss / (len(train_loader) * 4)

    epoch_loss_H_II /= len(train_loader)
    epoch_loss_T /= len(train_loader)
    epoch_loss_He_II /= len(train_loader)
    epoch_loss_He_III /= len(train_loader)

    return train_loss, epoch_loss_H_II, epoch_loss_T, epoch_loss_He_II, epoch_loss_He_III


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):
    """
    Main function for the CMLP run script

    Args:
        config: user config object

    Returns:
        Nothing
    """

    data_products_path = run_setup_run(config)

    train_loader, val_loader, test_loader = run_get_data_loaders_four_profiles(config)

    # -----------------------------------------------------------------
    # initialise model + move to GPU if CUDA present
    # -----------------------------------------------------------------
    model = CMLP(config, device)
    print('\n\tusing model CMLP\n')

    if cuda:
        model.cuda()

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    # -----------------------------------------------------------------
    # book keeping arrays
    # -----------------------------------------------------------------
    avg_train_loss_array = avg_val_loss_mse_array = avg_val_loss_dtw_array = np.empty(0)
    combined_train_loss_array = combined_val_loss_mse_array = combined_val_loss_dtw_array = np.empty((0, 4))

    # -----------------------------------------------------------------
    # keep the model with min validation loss
    # -----------------------------------------------------------------
    best_model = copy.deepcopy(model)
    best_loss_dtw = best_loss_mse = np.inf
    best_epoch_mse = best_epoch_dtw = 0

    # -----------------------------------------------------------------
    # Early Stopping Criteria
    # -----------------------------------------------------------------
    n_epoch_without_improvement = 0
    stopped_early = False
    epochs_trained = -1

    # -----------------------------------------------------------------
    # FORCE STOPPING
    # -----------------------------------------------------------------
    global FORCE_STOP
    FORCE_STOP = False
    if FORCE_STOP_ENABLED:
        signal.signal(signal.SIGINT, force_stop_signal_handler)
        print('\n Press Ctrl + C to stop the training anytime and exit while saving the results.\n')

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):

       # training
        (train_loss,
         epoch_loss_H_II,
         epoch_loss_T,
         epoch_loss_He_II,
         epoch_loss_He_III) = cmlp_train(model, optimizer, train_loader, config)

        # book keeping
        avg_train_loss_array = np.append(avg_train_loss_array, train_loss)

        stacked_train_loss = np.stack((epoch_loss_H_II, epoch_loss_T, epoch_loss_He_II, epoch_loss_He_III))

        combined_train_loss_array = np.concatenate((combined_train_loss_array,
                                                    stacked_train_loss.reshape(1, -1)), axis=0)

        # validation
        (val_loss_mse,
         val_loss_dtw,
         stacked_loss_mse,
         stacked_loss_dtw) = cmlp_run_evaluation(current_epoch=epoch,
                                                 data_loader=val_loader,
                                                 model=model,
                                                 path=data_products_path,
                                                 config=config,
                                                 print_results=False,
                                                 save_results=False,
                                                 best_model=False
                                                 )

        avg_val_loss_mse_array = np.append(avg_val_loss_mse_array, val_loss_mse)
        avg_val_loss_dtw_array = np.append(avg_val_loss_dtw_array, val_loss_dtw)

        combined_val_loss_mse_array = np.concatenate((combined_val_loss_mse_array,
                                                      stacked_loss_mse.reshape(1, -1)), axis=0)
        combined_val_loss_dtw_array = np.concatenate((combined_val_loss_dtw_array,
                                                      stacked_loss_dtw.reshape(1, -1)), axis=0)

        # save the best performing model
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

        print("[Epoch %d/%d] [Train loss {}}: {}] [Validation loss MSE: {}] [Validation loss DTW: {}] "
              "[Best_epoch (mse): {}] [Best_epoch (dtw): {}]"
              .format(epoch, config.n_epochs, config.loss_type, train_loss, val_loss_mse, val_loss_dtw,
                      best_epoch_mse, best_epoch_dtw)
              )

        if FORCE_STOP or (EARLY_STOPPING and n_epoch_without_improvement >= EARLY_STOPPING_THRESHOLD_CMLP):
            print("\033[96m\033[1m\nStopping Early\033[0m\n")
            stopped_early = True
            epochs_trained = epoch
            break

        if epoch % config.testing_interval == 0 or epoch == config.n_epochs:
            cmlp_run_evaluation(current_epoch=best_epoch_mse,
                                data_loader=test_loader,
                                model=best_model,
                                path=data_products_path,
                                config=config,
                                print_results=True,
                                save_results=True)

    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    # -----------------------------------------------------------------
    # save loss
    # ----------------------------------------------------------------
    utils_save_loss(combined_train_loss_array, data_products_path, config.profile_type, config.n_epochs, prefix='train')
    utils_save_loss(avg_train_loss_array, data_products_path, config.profile_type, config.n_epochs, prefix='train_avg')

    if config.loss_type == 'MSE':
        utils_save_loss(combined_val_loss_mse_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')
        utils_save_loss(avg_val_loss_mse_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val_avg')
    else:
        utils_save_loss(combined_val_loss_dtw_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')
        utils_save_loss(avg_val_loss_dtw_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val_avg')

    # -----------------------------------------------------------------
    # Evaluate the best model by using the test set
    # -----------------------------------------------------------------
    (best_test_mse,
     best_test_dtw,
     stacked_test_loss_mse,
     stacked_test_loss_dtw) = cmlp_run_evaluation(best_epoch_mse,
                                                  test_loader,
                                                  best_model,
                                                  data_products_path,
                                                  config,
                                                  print_results=True,
                                                  save_results=True,
                                                  best_model=True
                                                  )

    # -----------------------------------------------------------------
    # Save the best model and the final model
    # -----------------------------------------------------------------
    utils_save_model(best_model.state_dict(), data_products_path, 'C', best_epoch_mse, best_model=True)

    # -----------------------------------------------------------------
    # Save some results to config object for later use
    # -----------------------------------------------------------------
    config.best_epoch = best_epoch_mse
    config.best_epoch_mse = best_epoch_mse
    config.best_epoch_dtw = best_epoch_dtw

    config.best_val_mse = best_loss_mse
    config.best_val_dtw = best_loss_dtw

    config.best_test_mse = best_test_mse
    config.best_test_mse_H_II = stacked_test_loss_mse[0]
    config.best_test_mse_T = stacked_test_loss_mse[1]
    config.best_test_mse_He_II = stacked_test_loss_mse[2]
    config.best_test_mse_He_III = stacked_test_loss_mse[3]

    config.best_test_dtw = best_test_dtw
    config.best_test_dtw_H_II = stacked_test_loss_dtw[0]
    config.best_test_dtw_T = stacked_test_loss_dtw[1]
    config.best_test_dtw_He_II = stacked_test_loss_dtw[2]
    config.best_test_dtw_He_III = stacked_test_loss_dtw[3]

    config.stopped_early = stopped_early
    config.epochs_trained = epochs_trained
    config.early_stopping_threshold = EARLY_STOPPING_THRESHOLD_CMLP

    utils_save_config_to_log(config)  # Overwrite config object
    utils_save_config_to_file(config)

    if config.analysis:
        print("\n\033[96m\033[1m\nRunning analysis\033[0m\n")

        analysis_loss_plot(config)
        analysis_auto_plot_profiles(config, k=30, prefix='best')
        analysis_parameter_space_plot(config, prefix='best')

    # -----------------------------------------------------------------
    # End of main
    # -----------------------------------------------------------------


def cmlp_input_sanity_checks(config):
    """
    Perform user input checks. Print help and exit if anything is wrong

    Args:
        config: user config object

    Returns:
        Nothing

    """

    if config.data_dir is None:
        print('\nError: Parameter data_dir must not be empty. Exiting.\n')
        argparse.ArgumentParser().print_help()
        exit(1)

    if config.n_parameters not in [5, 8]:
        print('\nError: Number of parameters can currently only be either 5 or 8. Exiting.\n')
        argparse.ArgumentParser().print_help()
        exit(1)


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with neural networks (CMLP)')

    # arguments for data handling
    parser.add_argument('--data_dir', type=str, metavar='(string)', help='Path to data directory')

    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument("--testing_interval", type=int,
                        default=200, help="epoch interval between testing runs")

    parser.add_argument("--profile_len", type=int, default=1500,
                        help="number of profile grid points")

    parser.add_argument("--n_parameters", type=int, default=8,
                        help="number of RT parameters (5 or 8)")

    # network model switch
    parser.add_argument('--loss_type', type=str, default='MSE', metavar='(string)',
                        help='Pick a loss function: MSE (default) or DTW')

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=1500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches (default=32)")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate, default=0.0001")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    # use blow out filter?
    parser.add_argument("--filter_blowouts", dest='analysis', action='store_true',
                        help="use blowout filter on data set (default)")
    parser.add_argument("--no-filter_blowouts", dest='analysis', action='store_false',
                        help="do not use blowout filter on data set")
    parser.set_defaults(filter_blowouts=False)

    # cut parameter space
    parser.add_argument("--filter_parameters", dest='analysis', action='store_true',
                        help="use user_config to filter data set by parameters")
    parser.add_argument("--no-filter_parameters", dest='analysis', action='store_false',
                        help="do not use user_config to filter data set by parameters (default)")
    parser.set_defaults(filter_parameters=False)

    # analysis
    parser.add_argument("--analysis", dest='analysis', action='store_true',
                        help="automatically generate some plots (default)")
    parser.add_argument("--no-analysis", dest='analysis', action='store_false', help="do not run analysis")
    parser.set_defaults(analysis=True)

    my_config = parser.parse_args()

    # additional settings
    my_config.profile_type = 'C'
    my_config.model = 'CMLP'
    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.data_dir = os.path.abspath(my_config.data_dir)

    cmlp_input_sanity_checks(my_config)

    my_config = run_set_parameter_limits(my_config)

    utils_print_config_object(my_config)

    main(my_config)
