import torch
import torch.nn as nn
import numpy as np
import os.path as osp

from models.mlp import *
from models.cvae import *
from models.cgan import *
from models.lstm import *
from models.cmlp import *
from models.clstm import *

from common.utils import *
from common.settings import *
from common.clock import Clock
from common.plot import plot_inference_profiles
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
#  LSTM
# -----------------------------------------------------------------
def inference_lstm(run_dir, model_file_name, parameters, profile_type, measure_time=False):
    """
    Function to use user specified parameters with the LSTM in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the LSTM')

    config = utils_load_config(run_dir)

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

        # -----------------------------------------------------------------
    # initialise model + check for CUDA
    # -----------------------------------------------------------------
    if config.model == 'LSTM1':
        model = LSTM1(config, device)
    else:
        print('Error. Check if you are using the right model. Exiting.')
        exit(1)

    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    model.load_state_dict(torch.load(model_path))
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
#  CGAN
# -----------------------------------------------------------------
def inference_cgan(run_dir, model_file_name, parameters, profile_type, measure_time=False):
    """
    Function to use user specified parameters with the CGAN
    trained generator in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the CGAN generator')

    config = utils_load_config(run_dir)
    batch_size = np.shape(parameters)[0]
    
    # set up parameters
    if config.n_parameters == 5:
        parameter_limits = sp.p5_limits

    if config.n_parameters == 8:
        parameter_limits = sp.p8_limits

    if SCALE_PARAMETERS:
        parameters = utils_scale_parameters(limits=parameter_limits, parameters=parameters)
    
    # configure generator input
    parameters = torch.from_numpy(parameters).to(device)
    parameters = Variable(parameters.type(FloatTensor))
    latent_vector = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim)))).to(device)

    if config.gen_model == 'GEN2':
        generator = Generator2(config)
    elif config.gen_model == 'GEN1':
        generator = Generator1(config)
    else:
        print('Error. Check if you are using the right model. Exiting.')
        exit(1)

    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    generator.load_state_dict(torch.load(model_path))
    generator.to(device)
    generator.eval()
    
    output = {}
    output_time = None
    
    # run inference
    with torch.no_grad():
        gen_profile = generator(latent_vector, parameters)
        output[profile_type] = gen_profile
    
    if measure_time:
        if cuda:
            output_time = {}
            clock = Clock()
            avg_time, std_time = clock.get_time(generator, [latent_vector, parameters])
            output_time['avg_time'] = avg_time
            output_time['std_time'] = std_time
        else:
            print('\nTime can only be measured when on GPU. skipping. \n')

    return output, output_time


# -----------------------------------------------------------------
#  CVAE
# -----------------------------------------------------------------
def inference_cvae(run_dir, model_file_name, parameters, profile_type, measure_time=False):
    """
    Function to use user specified parameters with the CVAE in inference mode.

    Returns an array of one or more generated profiles
    """
    print('Running inference for the CVAE (decoder)')

    config = utils_load_config(run_dir)

    batch_size = np.shape(parameters)[0]
    
    latent_vector = np.zeros((batch_size, config.latent_dim))
    # all zeros for now ... closest to unit Gaussian

    # set up parameters
    if config.n_parameters == 5:
        parameter_limits = sp.p5_limits

    if config.n_parameters == 8:
        parameter_limits = sp.p8_limits

    if SCALE_PARAMETERS:
        parameters = utils_scale_parameters(limits=parameter_limits, parameters=parameters)

    # convert numpy arrays to tensors
    parameters = torch.from_numpy(parameters)
    latent_vector = torch.from_numpy(latent_vector)

    parameters = Variable(parameters.type(FloatTensor))
    latent_vector = Variable(latent_vector.type(FloatTensor))

    # concatenate both vector, i.e. condition the latent vector with the parameters
    cond_z = torch.cat((latent_vector, parameters), 1)
     
    # prepare the model
    if config.model == 'CVAE1':
        model = CVAE1(config)
    else:
        print('Error. Check if you are using the right model. Exiting.')
        exit(1)
    
    # move model and input to the available device
    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    cond_z.to(device)
    model.eval()

    output = {}
    output_time = None
   
    # run inference
    with torch.no_grad():
        profile_gen = model.decode(cond_z)
        output[profile_type] = profile_gen
        
    if measure_time:
        if cuda:
            output_time = {}
            clock = Clock()
            avg_time, std_time = clock.get_time(model.decode, [cond_z])
            output_time['avg_time'] = avg_time
            output_time['std_time'] = std_time
        else:
            print('\nTime can only be measured when on GPU. skipping. \n')

    return output, output_time


# -----------------------------------------------------------------
#  MLP
# -----------------------------------------------------------------
def inference_mlp(run_dir, model_file_name, parameters, profile_type, measure_time=False):
    """
    Function to use user specified parameters with the MLP in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the MLP')

    config = utils_load_config(run_dir)

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

    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    model.load_state_dict(torch.load(model_path))
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
#  CMLP
# -----------------------------------------------------------------
def inference_cmlp(run_dir, model_file_name, parameters, profile_type, measure_time=False):
    """
    Function to use user specified parameters with the CMLP in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the CMLP')

    config = utils_load_config(run_dir)

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

    model = CMLP(config, device)
    
    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    output = {}
    output_time = None

    # run inference
    with torch.no_grad():
        gen_profile_H, gen_profile_T, gen_profile_He_II, gen_profile_He_III = model(parameters)
        output['H'] = gen_profile_H
        output['T'] = gen_profile_T
        output['He_II'] = gen_profile_He_II
        output['He_III'] = gen_profile_He_III

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
#  CLSTM
# -----------------------------------------------------------------
def inference_clstm(run_dir, model_file_name, parameters, profile_type, measure_time=False):
    """
    Function to use user specified parameters with the CLSTM in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the CLSTM')

    config = utils_load_config(run_dir)

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

    model = CLSTM(config, device)
    
    model_path = osp.join(run_dir, DATA_PRODUCTS_DIR, model_file_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    output = {}
    output_time = None

    # run inference
    with torch.no_grad():
        gen_profile_H, gen_profile_T, gen_profile_He_II, gen_profile_He_III = model(parameters)
        output['H'] = gen_profile_H
        output['T'] = gen_profile_T
        output['He_II'] = gen_profile_He_II
        output['He_III'] = gen_profile_He_III

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
#  functions for test runs
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
        print('Inference time for %s: %e±%e ms' % ('MLP', output_time['avg_time'], output_time['std_time']))

    # save, plot etc

    # TODO: utils_save_single_profile(profile, path, file_name)
    # plots should be done elsewhere by hand


def inference_model_comparison():
    """
    Function to generate inference profiles using all architectures,
    plot those and compare the inference time.
    """
    
    # MLP test
    mlp_run_dir = './test/MLP_H_run/'
    mlp_model_file_name = 'best_model_H_1105_epochs.pth.tar'
    
    # CVAE test
    cvae_run_dir = './test/paper/run_CVAE1_DTW_31H'
    cvae_model_file_name = 'best_model_H_3558_epochs.pth.tar'

    # CGAN test
    cgan_run_dir = './test/paper/run_CGAN_MSE_48H'
    cgan_model_file_name = 'best_model_H_12774_epochs.pth.tar'

    # LSTM test
    lstm_run_dir = './test/paper/run_LSTM1_DTW_30H'
    lstm_model_file_name = 'best_model_H_338_epochs.pth.tar'

    # CMLP test
    cmlp_run_dir = './test/paper/run_CMLP_MSE_47C/'
    cmlp_model_file_name = 'best_model_C_910_epochs.pth.tar'
     
    # CLSTM test
    clstm_run_dir = './test/paper/run_CLSTM1_MSE_7C/'
    clstm_model_file_name = 'best_model_C_242_epochs.pth.tar'
   
    p = np.zeros((8))  # has to be 2D array because of BatchNorm

    p[0] = 8.825165  # M_halo
    p[1] = 8.285341  # redshift
    p[2] = 14.526998  # source Age
    p[3] = 1.491899   # qsoAlpha
    p[4] = 0.79072833   # qsoEfficiency
    p[5] = 0.48244837  # starsEscFrac
    p[6] = 1.5012491  # starsIMFSlope
    p[7] = 1.5323509  # starsIMFMassMinLog

    p_2D = p.copy()
    p_2D = p_2D[np.newaxis, :]
    
    output_mlp, output_time_mlp = inference_mlp(mlp_run_dir, mlp_model_file_name, p_2D.copy(), 'H', measure_time=True)
    if output_time_mlp is not None:
        print('\tInference time for %s: %e±%e ms\n' % ('MLP', output_time_mlp['avg_time'], output_time_mlp['std_time']))
    
    output_cvae, output_time_cvae = inference_cvae(cvae_run_dir, cvae_model_file_name, p_2D.copy(), 'H', measure_time=True)
    if output_time_cvae is not None:
        print('\tInference time for %s: %e±%e ms\n' % ('CVAE', output_time_cvae['avg_time'], output_time_cvae['std_time']))
    
    output_cgan, output_time_cgan = inference_cgan(cgan_run_dir, cgan_model_file_name, p_2D.copy(), 'H', measure_time=True)
    if output_time_cgan is not None:
        print('\tInference time for %s: %e±%e ms\n' % ('CGAN', output_time_cgan['avg_time'], output_time_cgan['std_time']))

    output_lstm, output_time_lstm = inference_lstm(lstm_run_dir, lstm_model_file_name, p_2D.copy(), 'H', measure_time=False)
    if output_time_lstm is not None:
        print('\tInference time for %s: %e±%e ms\n' % ('LSTM', output_time_lstm['avg_time'], output_time_lstm['std_time']))

    output_cmlp, output_time_cmlp = inference_cmlp(cmlp_run_dir, cmlp_model_file_name, p_2D.copy(), 'H', measure_time=True)
    if output_time_cmlp is not None:
        print('\tInference time for %s: %e±%e ms\n' % ('CMLP', output_time_cmlp['avg_time'], output_time_cmlp['std_time']))
    
    output_clstm, output_time_clstm = inference_clstm(clstm_run_dir, clstm_model_file_name, p_2D.copy(), 'H', measure_time=False)
    if output_time_clstm is not None:
        print('\tInference time for %s: %e±%e ms\n' % ('CLSTM', output_time_clstm['avg_time'], output_time_clstm['std_time']))
    
    profiles = torch.cat((output_mlp['H'], output_cvae['H'], output_cgan['H'], output_lstm['H'],
                          output_cmlp['H'], output_clstm['H']), dim=0).cpu().numpy() 
    plot_inference_profiles(profiles, 'H', p, output_dir='./', labels=['MLP','CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'])

    
# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    # inference_test_run_mlp()
    inference_model_comparison()