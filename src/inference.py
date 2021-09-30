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
def inference_lstm(parameters, profile_type, pretrained_models_dir, model_file_name = None, 
                   config_file_name=None, measure_time=False):
    """
    Function to use user specified parameters with the LSTM in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the LSTM')

    if model_file_name is None:
        model_file_name = 'best_model_%s_LSTM.pth.tar'%(profile_type)
    if config_file_name is None:
        config_file_name = 'config_%s_LSTM.dict'%(profile_type)
    
    model_path = osp.join(pretrained_models_dir, model_file_name)    
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)

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
def inference_cgan(parameters, profile_type, pretrained_models_dir, model_file_name = None, 
                   config_file_name=None, measure_time=False):
    """
    Function to use user specified parameters with the CGAN
    trained generator in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the CGAN generator')

    if model_file_name is None:
        model_file_name = 'best_model_%s_CGAN.pth.tar'%(profile_type)
    if config_file_name is None:
        config_file_name = 'config_%s_CGAN.dict'%(profile_type)
    
    model_path = osp.join(pretrained_models_dir, model_file_name)    
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)
    
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
def inference_cvae(parameters, profile_type, pretrained_models_dir, model_file_name=None, 
                   config_file_name=None, measure_time=False):
    """
    Function to use user specified parameters with the CVAE in inference mode.

    Returns an array of one or more generated profiles
    """
    print('Running inference for the CVAE (decoder)')
    
    if model_file_name is None:
        model_file_name = 'best_model_%s_CVAE.pth.tar'%(profile_type)
    if config_file_name is None:
        config_file_name = 'config_%s_CVAE.dict'%(profile_type)
    
    model_path = osp.join(pretrained_models_dir, model_file_name)    
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)

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
def inference_mlp(parameters, profile_type, pretrained_models_dir, model_file_name=None, 
                   config_file_name=None, measure_time=False):
    """
    Function to use user specified parameters with the MLP in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the MLP')

    if model_file_name is None:
        model_file_name = 'best_model_%s_MLP.pth.tar'%(profile_type)
    if config_file_name is None:
        config_file_name = 'config_%s_MLP.dict'%(profile_type)
    
    model_path = osp.join(pretrained_models_dir, model_file_name)    
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)

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
def inference_cmlp(parameters, profile_type, pretrained_models_dir, model_file_name='best_model_C_CMLP.pth.tar', 
                   config_file_name='config_C_CMLP.dict', measure_time=False):
    """
    Function to use user specified parameters with the CMLP in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the CMLP')
    
    model_path = osp.join(pretrained_models_dir, model_file_name)    
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)

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
def inference_clstm(parameters, profile_type, pretrained_models_dir, model_file_name='best_model_C_CLSTM.pth.tar', 
                   config_file_name='config_C_CLSTM.dict', measure_time=False):
    """
    Function to use user specified parameters with the CLSTM in inference mode.

    Returns an array of one or more generated profiles
    """

    print('Running inference for the CLSTM')
    
    model_path = osp.join(pretrained_models_dir, model_file_name)    
    config = utils_load_config(pretrained_models_dir, file_name=config_file_name)

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
#  functions for test runs - MLP
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
        print('Inference time for %s: %e ± %e ms' % ('MLP', output_time['avg_time'], output_time['std_time']))

    # save, plot etc
    # TODO: utils_save_single_profile(profile, path, file_name)


# -----------------------------------------------------------------
#  functions for test runs - CMLP
# -----------------------------------------------------------------
def inference_test_run_cmlp():
    """
    Example function to demonstrate how to use the fully trained CMLP with custom parameters.

    Returns nothing so far
    """

    # CMLP test
    cmlp_run_dir = './production_CMLP_MSE/run_2021_09_29__00_37_07/'
    cmlp_model_file_name = 'best_model_C_1378_epochs.pth.tar'

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

    output_cmlp, output_time_cmlp = inference_cmlp(cmlp_run_dir,
                                                   cmlp_model_file_name,
                                                   p_2D.copy(), 'H', measure_time=True)

    if output_time_cmlp is not None:
        print('\tInference time for %s: %e ± %e ms\n' % ('CMLP', output_time_cmlp['avg_time'], output_time_cmlp['std_time']))


# -----------------------------------------------------------------
#  functions for test runs - overall model comparison
# -----------------------------------------------------------------
def inference_model_comparison(pretrained_models_dir, profile_type, actual_parameters, actual_profiles=None,
                               models_to_use=['MLP','CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'],
                               measure_time=False, plot_output_dir='./', prefix=None):
    """
    Function to generate inference profiles using all architectures,
    plot them and compare the inference time.
    
    profile_type: type of profile you want to plot
    actual_parameters: parameters of shape (batch_size, parameters) 
                       or (parameters) for which inference is to be run
    actual_profile: actual profile corresponding to each parameter if known 
                    of shape (batch_size, profile_len) or (profile_len) for which inference is to be run
    models_to_use: list of pretrained models to be used for inference
                    (default: ['MLP','CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'])
    measure_time: whether to measure inference time for each model while running inference or not
    plot_output_dir: directory where inference plots will be placed (default: current_directory)
    prefix: prefix to be used in the name of the plots 
    """
    # model to corresponding function mapping     
    model_to_func_dict = {
        'MLP':inference_mlp,
        'CVAE':inference_cvae,
        'CGAN':inference_cgan,
        'LSTM':inference_lstm,
        'CMLP':inference_cmlp,
        'CLSTM':inference_clstm
    }

    # convert input to 2D form     
    p_2D = actual_parameters.copy()
    if len(np.shape(p_2D)) != 2:
        p_2D = p_2D[np.newaxis, :]
    
    # flag if actual profile correspoding to parameters is known
    ACTUAL_PROFILE = actual_profiles is not None
    
     
    # number of parameters to run inference for    
    batch_size = np.shape(p_2D)[0]

    # array of labels for profiles to be plotted finally    
    labels = []
    # array of profiles to be plotted finally    
    profiles = torch.zeros([batch_size, 0, 1500], device=device)
    
    # if actual profiles are known
    if ACTUAL_PROFILE:
        actual_profiles_2D = actual_profiles.copy()
        
        # convert them to 2D array
        if len(np.shape(actual_profiles_2D)) != 2:
            actual_profiles_2D = actual_profiles_2D[np.newaxis, :]
        
        # and concat them to profiles to plot
        actual_profiles_2D = torch.from_numpy(actual_profiles_2D).to(device)
        profiles = torch.cat((profiles, torch.unsqueeze(actual_profiles_2D, 1)), dim=1)
        labels.append('Simulation')

    # compute profiles for the parameters using the specified pre_trained models
    # and concat the output profiles with profiles to plot 
    for model in models_to_use:
        output_profile, output_time = model_to_func_dict[model](p_2D.copy(), profile_type,
                                                                pretrained_models_dir, measure_time=measure_time)
        if output_time is not None:
            print('\tInference time for %s: %e±%e ms\n' % (model, output_time['avg_time'], output_time['std_time']))
        
        profiles = torch.cat((profiles, torch.unsqueeze(output_profile[profile_type], 1)), 1)
        labels.append(model)
        
    # move the profiles to cpu and convert to numpy array     
    profiles = profiles.cpu().numpy()
    
    for i in range(batch_size):
        # if prefix is specified, make it unique for each profile in the batch 
        # else if prefix is None, timestamp will be used as prefix        
        if prefix is not None:
            prefix = prefix+str(i+1)
        
        # plot the profiles    
        plot_inference_profiles(profiles[i], profile_type, p_2D[i], output_dir=plot_output_dir,
                                labels=labels, prefix=prefix)
        
    
def inference_main(paper_data_directory,
                   pretrained_models_dir=None,
                   models_to_use=['MLP','CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'],
                   measure_time=False):
    """ 
    Function to load the sde data and run it through the 
    inference_model_comparison function for H and T profiles.
    """
    
    base_path = osp.join(paper_data_directory, ARCH_COMPARISON_DIR, SD_RUNS_DIR)
    
    inference_plots_path = osp.join(paper_data_directory, ARCH_COMPARISON_DIR, INFERENCE_DIR)    
    # Create inference plots dir if doesn't exist
    utils_create_output_dirs([inference_plots_path])
    
    if pretrained_models_dir is None:
        pretrained_models_dir = osp.join(paper_data_directory, PRETRAINED_MODELS_DIR)

    for i in range(1,4):
        # path of actual profiles
        parameter_file_path = osp.join(base_path, 'run_%d'%(i), 'run_%d_parameters.npy'%(i))
        H_profile_path = osp.join(base_path, 'run_%d'%(i), 'run_%d_profile_HII.npy'%(i))
        T_profile_path = osp.join(base_path, 'run_%d'%(i), 'run_%d_profile_T.npy'%(i))
        He_II_profile_path = osp.join(base_path, 'run_%d'%(i), 'run_%d_profile_HeII.npy'%(i))
        He_III_profile_path = osp.join(base_path, 'run_%d'%(i), 'run_%d_profile_HeIII.npy'%(i))
        
        # load the actual profiles
        parameters = np.load(parameter_file_path)
        H_II_profiles = np.load(H_profile_path)
        T_profiles = np.load(T_profile_path)
        He_II_profiles = np.load(He_II_profile_path)
        He_III_profiles = np.load(He_III_profile_path)
        
        if USE_LOG_PROFILES:
            # add a small number to avoid trouble
            H_II_profiles = np.log10(H_II_profiles + 1.0e-6)
            He_II_profiles = np.log10(He_II_profiles + 1.0e-6)
            He_III_profiles = np.log10(He_III_profiles + 1.0e-6)
            T_profiles = np.log10(T_profiles)

        inference_model_comparison(pretrained_models_dir, 'H', parameters, actual_profiles=H_II_profiles,
                                   models_to_use=models_to_use,
                                   plot_output_dir=inference_plots_path, prefix='run_%d'%(i), measure_time=measure_time)
        inference_model_comparison(pretrained_models_dir, 'T', parameters, actual_profiles=T_profiles,
                                   models_to_use=models_to_use,
                                   plot_output_dir=inference_plots_path, prefix='run_%d'%(i), measure_time=measure_time)
        

# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    paper_data_directory = '../paper_data/'
    models_to_use=['MLP','CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM']
    pretrained_models_dir = osp.join(paper_data_directory, PRETRAINED_MODELS_DIR)
#     inference_main(paper_data_directory,
#                    pretrained_models_dir=pretrained_models_dir,
#                    models_to_use=models_to_use,
#                    measure_time=False)
#     inference_test_run_cmlp()
    
    # To have a custom run without knowing actual profile    
    p = np.zeros((8))  # has to be 2D array because of BatchNorm
    p[0] = 8.825165  # M_halo
    p[1] = 8.285341  # redshift
    p[2] = 14.526998  # source Age
    p[3] = 1.491899   # qsoAlpha
    p[4] = 0.79072833   # qsoEfficiency
    p[5] = 0.48244837  # starsEscFrac
    p[6] = 1.5012491  # starsIMFSlope
    p[7] = 1.5323509  # starsIMFMassMinLog
    
    inference_model_comparison(
                        pretrained_models_dir=pretrained_models_dir,
                        profile_type='H',
                        actual_parameters=p,
                        actual_profiles=None,
                        models_to_use=models_to_use,
                        measure_time=False,
                        plot_output_dir='./',
                        prefix=None)
