# -----------------------------------------------------------------
# file and directory names
# -----------------------------------------------------------------
H_II_PROFILE_FILE = 'data_HII_profiles.npy'
T_PROFILE_FILE = 'data_T_profiles.npy'
He_II_PROFILE_FILE = 'data_HeII_profiles.npy'
He_III_PROFILE_FILE = 'data_HeIII_profiles.npy'
GLOBAL_PARAMETER_FILE = 'data_parameters.npy'

DATA_PRODUCTS_DIR = 'data_products'
PLOT_DIR = 'plots'
SD_RUNS_DIR = 'SD_runs'
INFERENCE_DIR = 'inference_plots'
PRETRAINED_MODELS_DIR = 'pretrained_models'
ARCH_COMPARISON_DIR = 'arch_comparison'

# -----------------------------------------------------------------
# settings for the data set
# -----------------------------------------------------------------
SPLIT_FRACTION = (0.80, 0.10, 0.10)  # training, validation, testing
SHUFFLE = True
SHUFFLE_SEED = 42

SCALE_PARAMETERS = True
USE_LOG_PROFILES = True

# -----------------------------------------------------------------
# run settings
# -----------------------------------------------------------------
EARLY_STOPPING = True
EARLY_STOPPING_THRESHOLD_CGAN = 3000
EARLY_STOPPING_THRESHOLD_LSTM = 60
EARLY_STOPPING_THRESHOLD_CLSTM = 60
EARLY_STOPPING_THRESHOLD_CVAE = 300
EARLY_STOPPING_THRESHOLD_CMLP = 200
EARLY_STOPPING_THRESHOLD_MLP = 200

FORCE_STOP_ENABLED = True
