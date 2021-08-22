import argparse
import copy
import signal

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from models.mlp import *
from common.dataset import RTdata
from common.filter import *
from common.utils import *
from common.analysis import *
import common.settings_parameters as sp
from common.settings import *

from common.soft_dtw_cuda import SoftDTW as SoftDTW_CUDA
from common.soft_dtw import SoftDTW as SoftDTW_CPU

# -----------------------------------------------------------------
#  global variables
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
#  loss function(s)
# -----------------------------------------------------------------
if cuda:
    soft_dtw_loss = SoftDTW_CUDA(use_cuda=True, gamma=0.1)
else:
    soft_dtw_loss = SoftDTW_CPU(use_cuda=False, gamma=0.1)


def mlp_loss_function(loss_function, gen_x, real_x, config):

    if loss_function == 'DTW':
        # profile tensors are of shape [batch size, profile length]
        # soft DTW expects input of shape [batch size, 1, profile length]
        if len(gen_x.size()) != 3:
            loss = soft_dtw_loss(gen_x.unsqueeze(1), real_x.view(-1, config.profile_len).unsqueeze(1)).mean()
        else:
            loss = soft_dtw_loss(gen_x, real_x.view(-1, config.profile_len)).mean()
    else:
        loss = F.mse_loss(input=gen_x, target=real_x.view(-1, config.profile_len), reduction='mean')

    return loss


def force_stop_signal_handler(sig, frame):
    global FORCE_STOP
    FORCE_STOP = True
    print("\033[96m\033[1m\nTraining will stop after this epoch. Please wait.\033[0m\n")


# -----------------------------------------------------------------
#   use MLP with test or validation set
# -----------------------------------------------------------------
def mlp_run_evaluation(current_epoch, data_loader, model, path, config,
                       print_results=False, save_results=False, best_model=False):
    """
    function runs the given dataset through the mlp model, returns mse_loss and dtw_loss,
    and saves the results as well as ground truth to file, if save_results is True.

    Args:
        current_epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
        print_results: print average loss to screen?
        save_results: flag to save generated profiles locally (default: False)
        best_model: flag for testing on best model
    """

    if save_results:
        print("\033[94m\033[1mTesting the MLP now at epoch %d \033[0m" % current_epoch)

    if cuda:
        model.cuda()

    if save_results:
        profiles_gen_all = torch.tensor([], device=device)
        profiles_true_all = torch.tensor([], device=device)
        parameters_true_all = torch.tensor([], device=device)

    # Note: ground truth data could be obtained elsewhere but by getting it from the data loader here
    # we don't have to worry about randomisation of the samples.

    model.eval()

    loss_dtw = 0.0
    loss_mse = 0.0

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):

            # configure input
            profiles_true = Variable(profiles.type(FloatTensor))
            parameters = Variable(parameters.type(FloatTensor))

            # inference
            profiles_gen = model(parameters)

            # compute loss via soft dtw
            # profile tensors are of shape [batch size, profile length]
            # soft dtw wants input of shape [batch size, 1, profile length]

            dtw = mlp_loss_function('DTW', profiles_true, profiles_gen, config)
            loss_dtw += dtw

            # compute loss via MSE:
            mse = mlp_loss_function('MSE', profiles_true, profiles_gen, config)
            loss_mse += mse

            if save_results:
                # collate data
                profiles_gen_all = torch.cat((profiles_gen_all, profiles_gen), 0)
                profiles_true_all = torch.cat((profiles_true_all, profiles_true), 0)
                parameters_true_all = torch.cat((parameters_true_all, parameters), 0)

    # mean of computed losses
    loss_mse = loss_mse / len(data_loader)
    loss_dtw = loss_dtw / len(data_loader)

    if print_results:
        print("results: MSE: %e DTW %e" % (loss_mse, loss_dtw))

    if save_results:
        # move data to CPU, re-scale parameters, and write everything to file
        profiles_gen_all = profiles_gen_all.cpu().numpy()
        profiles_true_all = profiles_true_all.cpu().numpy()
        parameters_true_all = parameters_true_all.cpu().numpy()

        parameters_true_all = utils_rescale_parameters(limits=parameter_limits, parameters=parameters_true_all)

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        utils_save_test_data(
            parameters=parameters_true_all,
            profiles_true=profiles_true_all,
            profiles_gen=profiles_gen_all,
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
    H_profile_file_path = utils_join_path(config.data_dir, H_II_PROFILE_FILE)
    T_profile_file_path = utils_join_path(config.data_dir, T_PROFILE_FILE)
    global_parameter_file_path = utils_join_path(config.data_dir, GLOBAL_PARAMETER_FILE)

    H_profiles = np.load(H_profile_file_path)
    T_profiles = np.load(T_profile_file_path)
    global_parameters = np.load(global_parameter_file_path)

    # -----------------------------------------------------------------
    # OPTIONAL: Filter (blow-out) profiles
    # -----------------------------------------------------------------
    if config.filter_blowouts:
        H_profiles, T_profiles, global_parameters = filter_blowout_profiles(H_profiles, T_profiles,
                                                                            global_parameters)

    if config.filter_parameters:
        global_parameters, [H_profiles, T_profiles] = filter_cut_parameter_space(global_parameters, [H_profiles, T_profiles])

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )

    # -----------------------------------------------------------------
    # book keeping arrays
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

        epoch_loss = 0

        # set model mode
        model.train()

        for i, (profiles, parameters) in enumerate(train_loader):

            # configure input
            real_profiles = Variable(profiles.type(FloatTensor))
            real_parameters = Variable(parameters.type(FloatTensor))

            # zero the gradients on each iteration
            optimizer.zero_grad()

            # generate a batch of profiles
            gen_profiles = model(real_parameters)

            loss = mlp_loss_function(config.loss_type, gen_profiles, real_profiles, config)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # end-of-epoch book keeping
        train_loss = epoch_loss / len(train_loader)    # divide by number of batches (!= batch size)
        train_loss_array = np.append(train_loss_array, train_loss)

        val_loss_mse, val_loss_dtw = mlp_run_evaluation(
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
            "[Epoch %d/%d] [Train loss %s: %e] [Validation loss MSE: %e] [Validation loss DTW: %e] "
            "[Best_epoch (mse): %d] [Best_epoch (dtw): %d]"
            % (epoch, config.n_epochs, config.loss_type, train_loss, val_loss_mse, val_loss_dtw,
               best_epoch_mse, best_epoch_dtw)
        )

        # check for testing criterion
        if epoch % config.testing_interval == 0 or epoch == config.n_epochs:

            best_test_mse, best_test_dtw = mlp_run_evaluation(best_epoch_mse, test_loader, best_model, data_products_path, config,
                                                              print_results=True, save_results=True, best_model=False)

        # early stopping check
        if FORCE_STOP or (EARLY_STOPPING and n_epoch_without_improvement >= EARLY_STOPPING_THRESHOLD_MLP):
            print("\033[96m\033[1m\nStopping Early\033[0m\n")
            stopped_early = True
            epochs_trained = epoch
            break

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

    # -----------------------------------------------------------------
    # Save the best model and the final model
    # -----------------------------------------------------------------
    utils_save_model(best_model.state_dict(), data_products_path, config.profile_type, best_epoch_mse, best_model=True)
    utils_save_model(model.state_dict(), data_products_path, config.profile_type, config.n_epochs, best_model=False)

    utils_save_loss(train_loss_array, data_products_path, config.profile_type, config.n_epochs, prefix='train')

    if config.loss_type == 'MSE':
        utils_save_loss(val_loss_mse_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')
    else:
        utils_save_loss(val_loss_dtw_array, data_products_path,
                        config.profile_type, config.n_epochs, prefix='val')

    # -----------------------------------------------------------------
    # Evaluate the best model by using the test set
    # -----------------------------------------------------------------
    best_test_mse, best_test_dtw = mlp_run_evaluation(best_epoch_mse, test_loader, best_model, data_products_path,
                                                      config, print_results=True, save_results=True, best_model=True)

    # -----------------------------------------------------------------
    # Save some results to config object for later use
    # -----------------------------------------------------------------
    setattr(config, 'best_epoch', best_epoch_mse)
    setattr(config, 'best_epoch_mse', best_epoch_mse)
    setattr(config, 'best_epoch_dtw', best_epoch_dtw)

    setattr(config, 'best_val_mse', best_loss_mse)
    setattr(config, 'best_val_dtw', best_loss_dtw)

    setattr(config, 'best_test_mse', best_test_mse)
    setattr(config, 'best_test_dtw', best_test_dtw)

    setattr(config, 'stopped_early', stopped_early)
    setattr(config, 'epochs_trained', epochs_trained)
    setattr(config, 'early_stopping_threshold', EARLY_STOPPING_THRESHOLD_MLP)

    # -----------------------------------------------------------------
    # Overwrite config object
    # -----------------------------------------------------------------
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # finished
    print('\nAll done!')

    # -----------------------------------------------------------------
    # Optional: analysis
    # -----------------------------------------------------------------
    if config.analysis:
        print("\n\033[96m\033[1m\nRunning analysis\033[0m\n")

        analysis_loss_plot(config)
        analysis_auto_plot_profiles(config, k=5, prefix='best')
        analysis_parameter_space_plot(config, prefix='best')


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with neural networks (MLP)')

    # arguments for data handling
    parser.add_argument('--data_dir', type=str, metavar='(string)', help='Path to data directory', required=True)

    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument("--testing_interval", type=int, default=100, help="epoch interval between testing runs")

    # physics related arguments
    parser.add_argument('--profile_type', type=str, default='H', metavar='(string)',
                        help='Select H for neutral hydrogen fraction or T for temperature profiles (default: H)')

    parser.add_argument("--profile_len", type=int, default=1500, help="number of profile grid points")

    parser.add_argument("--n_parameters", type=int, default=8, help="number of RT parameters (5 or 8)")

    # network model switch
    parser.add_argument('--model', type=str, default='MLP1', metavar='(string)',
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

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate, default=0.0002 ")

    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    parser.add_argument('--loss_type', type=str, default='MSE', metavar='(string)',
                        help='Pick a loss function: MSE (default) or DTW')

    # use blow out filter?
    parser.add_argument("--filter_blowouts", dest='analysis', action='store_true',
                        help="use blowout filter on data set (default)")
    parser.add_argument("--no-filter_blowouts", dest='analysis', action='store_false',
                        help="do not use blowout filter on data set")
    parser.set_defaults(filter_blowouts=True)

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
        parameter_limits = sp.p5_limits
        parameter_names_latex = sp.p5_names_latex

    if my_config.n_parameters == 8:
        parameter_limits = sp.p8_limits
        parameter_names_latex = sp.p8_names_latex

    if my_config.model not in ['MLP1', 'MLP2']:
        my_config.model = 'MLP1'

    if my_config.loss_type not in ['MSE', 'DTW']:
        my_config.model = 'MSE'

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.data_dir = os.path.abspath(my_config.data_dir)

    # run main program
    main(my_config)
