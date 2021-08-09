import argparse
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import copy

from models.cgan import *
from common.dataset import RTdata
from common.filter import *
from common.utils import *
from common.analysis import *
import common.parameter_settings as ps
from common.utils import utils_compute_dtw, utils_compute_mse, utils_save_model
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
#  loss function(s)
# -----------------------------------------------------------------
adversarial_loss = torch.nn.MSELoss()

if cuda:
    soft_dtw_loss = SoftDTW(use_cuda=True, gamma=0.1)
else:
    soft_dtw_loss = torch.nn.MSELoss()   # SoftDTW only works for cuda right now :-/


def cvae_mse_loss_function(gen_x, real_x, config):

    mse = F.mse_loss(input=gen_x, target=real_x.view(-1, config.profile_len), reduction='mean')

    return mse


# -----------------------------------------------------------------
#  parameter vector as generator input
# -----------------------------------------------------------------
def cgan_fake_parameters_gen_input(n_parameters, batch_size, global_parameters, mode=1):
    """
    function returns 2D numpy array, i.e. one vector of len n_parameters times batch_size

    Args:
        n_parameters: len of parameter vector
        batch_size: number of fake vectors needed
        global_parameters: numpy array containing all available parameters in the training + testing sets
        mode: 0 - use random numbers from [0, 1], which should not work
        mode: 1 - use randomly drawn p vectors from the whole data set
        mode: 2 - use randomly drawn parameters from the whole data set
    """

    if mode == 0:

        return np.random.rand(batch_size, n_parameters)

    elif mode == 1:

        a = np.zeros((batch_size, n_parameters))
        n_sample_total = len(global_parameters)

        for i in range(batch_size):

            # select a random row, i.e. parameter vector
            i_random = np.random.randint(0, n_sample_total)
            a[i] = global_parameters[i_random]

        return a

    else:

        a = np.zeros((batch_size, n_parameters))
        n_sample_total = len(global_parameters)

        for i in range(n_parameters):

            # global parameters array consists of columns of the different parameters, i.e.
            # a row is a full parameter vector. Here we use the columns separately:
            i_random_array = np.random.randint(0, high=n_sample_total, size=batch_size)

            a[:, i] = global_parameters[i_random_array, i]

        return a


# -----------------------------------------------------------------
#   use generator with test set
# -----------------------------------------------------------------
def cgan_run_test(epoch, data_loader, model, path, config, best_model=False):
    """
    This function runs the test data set through the generator, saves results as well as ground truth to file

    Args:
        epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: generator model
        config: config object with user supplied parameters
        best_model: flag for use with best model
    """

    print("\033[94m\033[1mTesting the Generator now at epoch %d \033[0m" % epoch)

    if cuda:
        model.cuda()

    test_profiles_gen_all = torch.tensor([], device=device)
    test_profiles_true_all = torch.tensor([], device=device)
    test_parameters_true_all = torch.tensor([], device=device)

    # Note: ground truth data could be obtained elsewhere but by getting it from the data loader
    # we don't have to worry about randomisation of the input data

    model.eval()

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):
            batch_size = profiles.shape[0]

            # latent vector: here we sample noise, TODO: use all zeros
            latent_vector = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))

            # configure input
            test_profiles_true = Variable(profiles.type(FloatTensor))
            test_parameters = Variable(parameters.type(FloatTensor))

            # inference
            test_profiles_gen = model(latent_vector, test_parameters)

            # collate data
            test_profiles_gen_all = torch.cat((test_profiles_gen_all, test_profiles_gen), dim=0)
            test_profiles_true_all = torch.cat((test_profiles_true_all, test_profiles_true), dim=0)
            test_parameters_true_all = torch.cat((test_parameters_true_all, test_parameters), dim=0)

    # move data to CPU, re-scale parameters, and write everything to file
    test_profiles_gen_all = test_profiles_gen_all.cpu().numpy()
    test_profiles_true_all = test_profiles_true_all.cpu().numpy()
    test_parameters_true_all = test_parameters_true_all.cpu().numpy()

    test_parameters_true_all = utils_rescale_parameters(limits=parameter_limits,
                                                        parameters=test_parameters_true_all)

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
        epoch=epoch,
        prefix=prefix
    )


# -----------------------------------------------------------------
# Evaluate generator on validation set
# -----------------------------------------------------------------
# def cgan_evaluate_generator(generator, data_loader, config):
#     """
#     This function runs the validation data set through the generator,
#     and computes mse and dtw on the predicted profiles and true profiles
#
#     Args:
#         generator: model that generates data and needs to be evaluated
#         data_loader: data loader used for the inference, most likely the validation set
#         config: config object with user supplied parameters
#     """
#
#     # set generator to evaluation mode (!Important)
#     generator.eval()
#
#     # Empty tensors to hold real and generated profiles
#     true_profiles_all = torch.empty((0, config.profile_len), device=device)
#     gen_profiles_all = torch.empty((0, config.profile_len), device=device)
#
#     with torch.no_grad():
#         for i, (profiles, parameters) in enumerate(data_loader):
#
#             # obtain batch size
#             batch_size = profiles.size()[0]
#
#             # configure input
#             latent_vector = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))
#             true_parameters = Variable(parameters.type(FloatTensor))
#             true_profiles = Variable(profiles.type(FloatTensor))
#
#             # obtain predictions from generator using real_parameters
#             gen_profiles = generator(latent_vector, true_parameters)
#
#             # append real and generated profiles to our tensor list
#             true_profiles_all = torch.cat((true_profiles_all, true_profiles), dim=0)
#             gen_profiles_all = torch.cat((gen_profiles_all, gen_profiles), dim=0)
#
#     # convert tensors to numpy arrays
#     real_profiles = true_profiles_all.cpu().numpy()
#     gen_profiles = gen_profiles_all.cpu().numpy()
#
#     # compute mse and dtw on numpy profiles
#     mse = utils_compute_mse(real_profiles, gen_profiles)
#     dtw = utils_compute_dtw(real_profiles, gen_profiles)
#
#     return mse, dtw


# -----------------------------------------------------------------
# Evaluate generator on validation set - NEW version
# -----------------------------------------------------------------
def cgan_evaluate_generator_new(generator, data_loader, config):
    """
    This function runs the validation data set through the generator,
    and computes MSE and DTW (Dynamic Time Warping), the latter if CUDA is available,
    on the predicted profiles and true profiles.

    Args:
        generator: model that generates data and needs to be evaluated
        data_loader: data loader used for the inference, most likely the validation set
        config: config object with user supplied parameters
    """

    # set generator to evaluation mode (!Important)
    generator.eval()

    val_loss_dtw = 0.0
    val_loss_mse = 0.0

    with torch.no_grad():
        for i, (profiles, parameters) in enumerate(data_loader):
            # obtain batch size
            batch_size = profiles.size()[0]

            # configure generator input
            latent_vector = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))
            true_parameters = Variable(parameters.type(FloatTensor))
            true_profiles = Variable(profiles.type(FloatTensor))

            # obtain predictions from generator using real_parameters
            gen_profiles = generator(latent_vector, true_parameters)

            # compute loss via soft dtw
            if cuda:

                # profile tensors are of shape [batch size, profile length]
                # soft dtw wants input of shape [batch size, 1, profile length]

                loss_dtw = soft_dtw_loss(true_profiles.unsqueeze(1), gen_profiles.unsqueeze(1))
                val_loss_dtw += loss_dtw.mean()

            # compute loss via MSE:
            loss_mse = cvae_mse_loss_function(true_profiles, gen_profiles, config)
            val_loss_mse += loss_mse.mean()

    val_loss_mse = val_loss_mse / len(data_loader)

    if not cuda:

        return val_loss_mse, val_loss_mse

    else:

        val_loss_dtw = val_loss_dtw / len(data_loader)
        return val_loss_mse, val_loss_dtw


# -----------------------------------------------------------------
#  Train Generator on a batch of dataset
# -----------------------------------------------------------------
def cgan_train_generator(generator, discriminator, optimizer, loss, global_parameters, batch_size, config):
    """
    This function runs the generator on a batch of data, and then optimizes it based on it's
    ability to fool the discriminator 

    Args:
        generator: model that is used to generate data
        discriminator: model that classifies data into generated and original classes
        optimizer: optimizer to be used for training
        loss: loss function
        global_parameters: 
        batch_size: no. of samples from dataset in the current iteration
        config: config object with user supplied parameters
    """

    # adversarial ground truths
    all_ones = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    
    # set generator to train mode
    generator.train()
    
    # zero the gradients on each iteration
    optimizer.zero_grad()

    # sample noise and parameters as generator input
    latent_vector = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))
    p = cgan_fake_parameters_gen_input(config.n_parameters,
                                       batch_size,
                                       global_parameters,
                                       mode=config.gen_parameter_mode)
    gen_parameters = Variable(FloatTensor(p))

    # generate a batch of profiles
    gen_profiles = generator(latent_vector, gen_parameters)

    # measures generator's ability to fool the discriminator
    validity = discriminator(gen_profiles, gen_parameters)
    gen_loss = loss(validity, all_ones)

    gen_loss.backward()
    optimizer.step()
    
    return gen_profiles, gen_parameters, gen_loss


# -----------------------------------------------------------------
#  Train discriminator on profiles generated by the generator
# -----------------------------------------------------------------
def cgan_train_discriminator(real_profiles, real_parameters, gen_profiles, gen_parameters,
                             discriminator, optimizer, loss, batch_size):
    """
    This function runs the discriminator on the profiles generated by the generator and on the real profiles, 
    and trains it to classify them into separate classes

    Args:
        real_profiles: actual profiles from dataset 
        real_parameters: parameters corresponding to real_profiles from the dataset
        gen_profiles: generated profiles from the generator
        gen_parameters: fake parameters used to generate the generated_profiles
        discriminator: model that learns to classify data into generated and original classes
        optimizer: optimizer to be used for training
        loss: loss function
        batch_size: no. of samples from dataset in the current iteration
    """

    # adversarial ground truths
    all_ones = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    all_zeros = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    discriminator.train()

    optimizer.zero_grad()

    # Loss for real profiles
    validity_real = discriminator(real_profiles, real_parameters)
    d_real_loss = loss(validity_real, all_ones)

    # Loss for fake profiles
    validity_fake = discriminator(gen_profiles.detach(), gen_parameters)
    d_fake_loss = loss(validity_fake, all_zeros)

    # Total discriminator loss
    dis_loss = (d_real_loss + d_fake_loss) / 2

    # d_loss.backward(retain_graph=True)
    dis_loss.backward()
    optimizer.step()
    
    return dis_loss


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
    # Filter (blow-out) profiles
    # -----------------------------------------------------------------
    if USE_BLOWOUT_FILTER:
        H_profiles, T_profiles, global_parameters = filter_blowout_profiles(H_profiles, T_profiles, global_parameters)

    # TODO: insert parameter space filter (see MLP)

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
    # select & initialise generator and discriminator + check for CUDA
    # -----------------------------------------------------------------
    if config.gen_model == 'GEN2':
        generator = Generator2(config)
    else:
        generator = Generator1(config)

    if config.dis_model == 'DIS2':
        discriminator = Discriminator2(config)
    else:
        discriminator = Discriminator1(config)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    # -----------------------------------------------------------------
    # book keeping arrays
    # -----------------------------------------------------------------
    train_loss_array_gen = np.empty(0)
    train_loss_array_dis = np.empty(0)

    best_mse = np.inf
    best_dtw = np.inf
    best_generator = None
    best_epoch_mse = 0
    best_epoch_dtw = 0

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs+1):

        epoch_loss_gen = 0
        epoch_loss_dis = 0

        for i, (profiles, parameters) in enumerate(train_loader):

            # configure input
            real_profiles = Variable(profiles.type(FloatTensor))
            real_parameters = Variable(parameters.type(FloatTensor))

            gen_profiles, gen_parameters, gen_loss = cgan_train_generator(
                generator=generator,
                discriminator=discriminator,
                optimizer=optimizer_G,
                loss=adversarial_loss,
                global_parameters=global_parameters,
                batch_size=profiles.shape[0],
                config=config
            )

            dis_loss = cgan_train_discriminator(
                real_profiles=real_profiles,
                real_parameters=real_parameters,
                gen_profiles=gen_profiles,
                gen_parameters=gen_parameters,
                discriminator=discriminator,
                optimizer=optimizer_D,
                loss=adversarial_loss,
                batch_size=profiles.shape[0]
            )

            epoch_loss_gen += gen_loss.item()
            epoch_loss_dis += dis_loss.item()

        # end-of-epoch book keeping
        average_loss_gen = epoch_loss_gen / len(train_loader.dataset)
        average_loss_dis = epoch_loss_dis / len(train_loader.dataset)

        train_loss_array_gen = np.append(train_loss_array_gen, average_loss_gen)
        train_loss_array_dis = np.append(train_loss_array_dis, average_loss_dis)

        mse_val, dtw_val = cgan_evaluate_generator_new(generator, val_loader, config)

        if mse_val < best_mse:
            best_mse = mse_val
            best_epoch_mse = epoch
            best_generator = copy.deepcopy(generator)

        if dtw_val < best_dtw:
            best_dtw = dtw_val
            best_epoch_dtw = epoch

        print(
            "[Epoch %d/%d] [Avg_disc_loss: %e] [Avg_gen_loss: %e] "
            "[Val_score: MSE: %e DTW %e] [Best_epoch (mse): %d] [Best_epoch (dtw): %d]"
            % (epoch, config.n_epochs, average_loss_dis, average_loss_gen,
               mse_val, dtw_val, best_epoch_mse, best_epoch_dtw)
        )

        # check for testing criterion
        if epoch % config.testing_interval == 0 or epoch == config.n_epochs:

            cgan_run_test(epoch, test_loader, generator, data_products_path, config)

    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    utils_save_model(best_generator.state_dict(), data_products_path, config.profile_type, best_epoch_mse)

    # save training stats
    utils_save_loss(train_loss_array_gen, data_products_path, config.profile_type, config.n_epochs, prefix='G_train')
    utils_save_loss(train_loss_array_dis, data_products_path, config.profile_type, config.n_epochs, prefix='D_train')

    # Fin
    print('\nAll done!')

    # -----------------------------------------------------------------
    # Optional: analysis
    # -----------------------------------------------------------------
    if config.analysis:

        print("\n\033[96m\033[1m\nRunning analysis\033[0m\n")

        analysis_loss_plot(config, gan=True)
        analysis_auto_plot_profiles(config, k=5, prefix='test')


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with neural networks (CGAN)')

    # arguments for data handling
    parser.add_argument('--data_dir', type=str, metavar='(string)', help='Path to data directory')
    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument("--testing_interval", type=int, default=100, help="epoch interval between testing runs")

    # physics related arguments
    parser.add_argument('--profile_type', type=str, default='H', metavar='(string)',
                        help='Select H for neutral hydrogen fraction or T for temperature profiles (default: H)')
    parser.add_argument("--profile_len", type=int, default=1500, help="number of profile grid points")
    parser.add_argument("--n_parameters", type=int, default=5, help="number of RT parameters (5 or 8)")

    parser.add_argument("--gen_parameter_mode", type=int, default=1, help="mode for generating fake parameters (0,1,2)")

    # network model switches
    parser.add_argument('--gen_model', type=str, default='GEN1', metavar='(string)',
                        help='Pick a generator model: GEN1 (default) or GEN2')

    parser.add_argument('--dis_model', type=str, default='DIS1', metavar='(string)',
                        help='Pick a discriminator model: DIS1 (default) or DIS2')

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches (default=32)")

    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true',
                        help="use batch normalisation in generator (default)")
    parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false',
                        help="Do not use batch normalisation in generator")
    parser.set_defaults(batch_norm=True)

    parser.add_argument("--dropout", dest='dropout', action='store_true',
                        help="Use dropout regularisation in discriminator (default)")
    parser.add_argument("--no-dropout", dest='dropout', action='store_false',
                        help="Do not use dropout regularisation in discriminator")
    parser.set_defaults(dropout=True)

    parser.add_argument("--dropout_value", type=float, default=0.25, help="dropout probability, default=0.25 ")

    parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space (default=10)")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate, default=0.0002 ")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

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

    if my_config.n_parameters != 5 and my_config.n_parameters != 8:
        print('\nError: Number of parameters can currently only be either 5 or 8. Exiting.\n')
        argparse.ArgumentParser().print_help()
        exit(1)

    if my_config.n_parameters == 5:
        parameter_limits = ps.p5_limits

    if my_config.n_parameters == 8:
        parameter_limits = ps.p8_limits

    if my_config.gen_model not in ['GEN1', 'GEN2']:
        my_config.model = 'GEN1'

    if my_config.dis_model not in ['DIS1', 'DIS2']:
        my_config.model = 'DIS1'

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.data_dir = os.path.abspath(my_config.data_dir)

    main(my_config)
