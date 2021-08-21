# -----------------------------------------------------------------
# file and directory names
# -----------------------------------------------------------------
H_PROFILE_FILE = 'data_Hprofiles.npy'
T_PROFILE_FILE = 'data_Tprofiles.npy'
HE1_PROFILE_FILE = 'data_Hprofiles.npy'
HE2_PROFILE_FILE = 'data_Tprofiles.npy'
GLOBAL_PARAMETER_FILE = 'data_parameters.npy'

DATA_PRODUCTS_DIR = 'data_products'
PLOT_DIR = 'plots'

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
EARLY_STOPPING_THRESHOLD_CGAN = 100
EARLY_STOPPING_THRESHOLD_LSTM = 40
EARLY_STOPPING_THRESHOLD_CVAE = 100
EARLY_STOPPING_THRESHOLD_MLP = 100

FORCE_STOP_ENABLED = True
