import numpy as np
import os.path as osp
import heapq
import sys; sys.path.append('..')
import csv
import os

from common.utils import utils_load_config
from common.plot import *
from common.settings import DATA_PRODUCTS_DIR, PLOT_DIR, PLOT_FILE_TYPE
from common.settings_parameters import p8_names_latex

# -----------------------------------------------------------------
# Purpose of the functions below is to automatically run the data
# products through the plotting routines (after a training run)
# and make sure the plots are placed in the corresponding PLOT_DIR
# directories.
# -----------------------------------------------------------------


def analysis_loss_plot(config, gan=False):
    """
    Function to load and plot the training and validation loss data and pass it to the plot function.

    Args:
        config:  user config object
        gan:  bool to indicate if a GAN was used?

    Returns:
        Nothing
    """

    data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
    plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    if not gan:
        if config.profile_type == 'C':
            train_loss_file = 'train_avg_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)
            val_loss_file = 'val_avg_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)
        else:
            train_loss_file = 'train_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)
            val_loss_file = 'val_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)

        loss_1 = np.load(osp.join(data_dir_path, train_loss_file))
        loss_2 = np.load(osp.join(data_dir_path, val_loss_file))
        loss_3 = None

    else:
        gen_loss_file = 'G_train_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)
        dis_loss_real_file = 'D_train_real_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)
        dis_loss_fake_file = 'D_train_fake_loss_%s_%d_epochs.npy' % (config.profile_type, config.n_epochs)

        loss_1 = np.load(osp.join(data_dir_path, gen_loss_file))
        loss_2 = np.load(osp.join(data_dir_path, dis_loss_real_file))
        loss_3 = np.load(osp.join(data_dir_path, dis_loss_fake_file))

    plot_loss_function(lf1=loss_1,
                       lf2=loss_2,
                       lf3=loss_3,
                       epoch=config.n_epochs,
                       lr=config.lr,
                       output_dir=plot_dir_path,
                       profile_type=config.profile_type,
                       file_type=PLOT_FILE_TYPE,
                       gan=gan
                       )


# -----------------------------------------------------------------
# Automatically plot test profiles
# -----------------------------------------------------------------
def analysis_auto_plot_profiles(config, k=5, base_path=None, prefix='test', epoch=None):
    """
    Function to load the test set inference results, compute the MSE using the ground truth, and pass a
    selected number of best and worst profiles to a plotting routing.

    Args:
        config: User config object
        k: number of best / worst examples to plot
        base_path: path to training run (can supersede path in config object  )
        prefix:  'best' or 'test'
        epoch: epoch of the output file

    Returns:
        Nothing

    """

    if base_path is not None:
        data_dir_path = osp.join(base_path, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(base_path, PLOT_DIR)
    else:
        data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    if prefix == 'test':
        epoch = epoch
    elif prefix == 'best':
        epoch = config.best_epoch

    parameter_true_file = prefix + ('_parameters_%s_%d_epochs.npy' % (config.profile_type, epoch))
    profiles_true_file = prefix + ('_profiles_true_%s_%d_epochs.npy' % (config.profile_type, epoch))
    profiles_gen_file = prefix + ('_profiles_gen_%s_%d_epochs.npy' % (config.profile_type, epoch))
    parameter_filename = prefix + ('_sample_parameters_%s_%d_epochs.txt' % (config.profile_type, epoch))

    # Model id from base_path
    model_id = osp.basename(base_path).split('_')[1:] if base_path else ["Unknown"]
    
    # 1. remove old parameter files
    parameter_filepath = osp.join(plot_dir_path, parameter_filename)
    if osp.exists(parameter_filepath):
        os.remove(parameter_filepath)

    parameters = np.load(osp.join(data_dir_path, parameter_true_file))
    profiles_true = np.load(osp.join(data_dir_path, profiles_true_file))
    profiles_gen = np.load(osp.join(data_dir_path, profiles_gen_file))

    # 2. compute MSE
    if config.profile_type == 'C':
        mse_array = (np.square((profiles_true) - (profiles_gen))).mean(axis=(2, 1))
    else:
        mse_array = (np.square((profiles_true) - (profiles_gen))).mean(axis=1)

    mse_array = np.log10(mse_array + 1e-11)

    # 3. find k lowest and largest MSE values and their respective indexes
    k_large_list = heapq.nlargest(k, range(len(mse_array)), mse_array.take)
    k_small_list = heapq.nsmallest(k, range(len(mse_array)), mse_array.take)

    # 4.  plot profiles for largest MSE
    print('Producing profile plot(s) for profiles with %d highest MSE' % k)

    for i in range(len(k_large_list)):
        index = k_large_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        tmp_parameters = parameters[index]
        tmp_profile_true = profiles_true[index]
        tmp_profile_gen = profiles_gen[index]

        plot_profile_single_new(profile_true=tmp_profile_true,
                            profile_inferred=tmp_profile_gen,
                            n_epoch=epoch,
                            output_dir=plot_dir_path,
                            profile_type=config.profile_type,
                            prefix=prefix,
                            sample_id=index,
                            parameter_filename=parameter_filename,
                            parameters=tmp_parameters,
                            file_type=PLOT_FILE_TYPE,
                            model_id = model_id
                            )

    # 5.  plot profiles for smallest MSE
    print('Producing profile plot(s) for profiles with %d lowest MSE' % k)

    for i in range(len(k_small_list)):
        index = k_small_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        tmp_parameters = parameters[index]
        tmp_profile_true = profiles_true[index]
        tmp_profile_gen = profiles_gen[index]

        plot_profile_single_new(profile_true=tmp_profile_true,
                            profile_inferred=tmp_profile_gen,
                            n_epoch=epoch,
                            output_dir=plot_dir_path,
                            profile_type=config.profile_type,
                            prefix=prefix,
                            sample_id=index,
                            parameter_filename=parameter_filename,
                            parameters=tmp_parameters,
                            file_type=PLOT_FILE_TYPE,
                            model_id=model_id
                            )


# -----------------------------------------------------------------
#  generate parameter space plots
# -----------------------------------------------------------------
def analysis_parameter_space_plot(config, base_path=None, prefix='test', epoch=None):
    """
     Function to load the test set inference results and pass them on to a plot function that visualises the
     error - parameter space relationship.

    Args:
        config: User config object
        base_path: path to training run (can supersede path in config object  )
        prefix:  'best' or 'test'
        epoch: epoch of the output file

    Returns:
        Nothing

    """

    print('Producing parameter space - MSE plot')

    # 1. read data
    if base_path is not None:
        data_dir_path = osp.join(base_path, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(base_path, PLOT_DIR)
    else:
        data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    if prefix == 'test':
        epoch = epoch
    elif prefix == 'best':
        epoch = config.best_epoch

    parameter_true_file = prefix + ('_parameters_%s_%d_epochs.npy' % (config.profile_type, epoch))
    profiles_true_file = prefix + ('_profiles_true_%s_%d_epochs.npy' % (config.profile_type, epoch))
    profiles_gen_file = prefix + ('_profiles_gen_%s_%d_epochs.npy' % (config.profile_type, epoch))

    parameters = np.load(osp.join(data_dir_path, parameter_true_file))
    profiles_true = np.load(osp.join(data_dir_path, profiles_true_file))
    profiles_gen = np.load(osp.join(data_dir_path, profiles_gen_file))

    plot_parameter_space_mse(parameters=parameters,
                             profiles_true=profiles_true,
                             profiles_gen=profiles_gen,
                             profile_type=config.profile_type,
                             n_epoch=epoch,
                             output_dir=plot_dir_path,
                             prefix=prefix,
                             file_type='png'   # pdf makes no sense here
                             )


# -----------------------------------------------------------------
#  generate error density plots
# -----------------------------------------------------------------
def analysis_error_density_plot(config, base_path=None, prefix='test', epoch=None, add_title=True):
    """
    Function to load the test set inference results and pass them on to a plot function that
    visualises the error distribution.

    Args:
        config: User config object
        base_path: path to training run (can supersede path in config object  )
        prefix:  'best' or 'test'
        epoch: epoch of the output file
        add_title: pass a plot title to the plot function

    Returns:
        Nothing
    """

    print('Producing error density - MSE plot')

    # 1. read data
    if base_path is not None:
        data_dir_path = osp.join(base_path, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(base_path, PLOT_DIR)
    else:
        data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    if prefix == 'test':
        epoch = epoch
        if epoch is None:
            print('Error: no argument provided for epoch to analysis_error_density_plot(). Exiting')
            exit(1)
    elif prefix == 'best':
        epoch = config.best_epoch

    profiles_true_file = prefix + ('_profiles_true_%s_%d_epochs.npy' % (config.profile_type, epoch))
    profiles_gen_file = prefix + ('_profiles_gen_%s_%d_epochs.npy' % (config.profile_type, epoch))

    profiles_true = np.load(osp.join(data_dir_path, profiles_true_file))
    profiles_gen = np.load(osp.join(data_dir_path, profiles_gen_file))

    plot_error_density_mse(profiles_true=profiles_true,
                           profiles_gen=profiles_gen,
                           n_epoch=epoch,
                           config=config,
                           output_dir=plot_dir_path,
                           prefix=prefix,
                           add_title=add_title,
                           file_type=PLOT_FILE_TYPE
                           )

# -----------------------------------------------------------------
#  generate latex table from csv file with parameters
# -----------------------------------------------------------------
def csv_to_latex_table(config, row_filter, base_path=None, prefix='best', headers=None, epoch=None):

    """
    Reads a CSV file and converts its content into a LaTeX table string,
    including column headers and filtering rows based on a list of integers.

    Args:
        config: User config object
        row_filter: List of integers to match the first column.
        base_path: path to training run (can supersede path in config object)
        prefix: 'best' or 'test'
        headers: Column headers in LaTeX format, separated by commas.
        epoch: epoch of the output file

    Returns:
        Nothing.
    """

    # Config
    if base_path is not None:
        plot_dir_path = osp.join(base_path, PLOT_DIR)
    else:
        plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    epoch = epoch if prefix == 'test' else config.best_epoch

    sample_parameters_path = f"{prefix}_sample_parameters_{config.profile_type}_{epoch}_epochs.txt"
    file_path = osp.join(plot_dir_path, sample_parameters_path)
    
    # Model id from base_path
    model_id = osp.basename(base_path).split('_')[1:] if base_path else ["Unknown"]

    # Remove old latex table file
    table_filepath = file_path.replace(".txt", "_latexTable.txt")
    if os.path.exists(table_filepath):
        os.remove(table_filepath)

    # Start the LaTeX table
    if headers is None:
        headers = p8_names_latex
    headers = ["Profile index"] + headers
    column_count = len(headers)
    latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{" + " c" * column_count + "}\n\\hline\n"
    
    # Add headers to the table
    latex_table += " & ".join(headers) + " \\\\\n\\hline\n"
    
    # Read the CSV file and add only the filtered rows to the table
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Check if the first column matches one of the values in row_filter
            if int(row[0]) in row_filter:
                table_row = [model_id[0] + " " + model_id[2]+ " " + row[0]]
                for r in row[1:]:
                    table_row.append("%.3f"%float(r))
                latex_table += " & ".join(table_row) + " \\\\\n\\hline\n"
    
    # Finish the LaTeX table
    latex_table += "\\end{tabular}\n\\caption{Table generated from CSV file}\n\\label{tab:my_table}\n\\end{table}"

    # Write out the latex table
    with open(table_filepath,'w') as file:
        file.write(latex_table)

# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there! Let\'s analyse some results\n')
    
    # Maybe add your own array here.
    # production_runs_path = '../test/production/production_runs/'
    production_runs_path = '../../paper_data/production_runs'

    best_runs = [
        # 'production_CGAN_MSE_H',
        # 'production_CGAN_MSE_T',
        'production_MLP_MSE_H',
        'production_MLP_MSE_T',
        'production_MLP_DTW_H',
        'production_MLP_DTW_T',
        # 'production_LSTM_MSE_H',
        # 'production_LSTM_MSE_T',
        # 'production_LSTM_DTW_H',
        # 'production_LSTM_DTW_T',
        # 'production_CVAE_MSE_H',
        # 'production_CVAE_MSE_T',
        # 'production_CVAE_DTW_H',
        # 'production_CVAE_DTW_T',
        # 'production_CMLP_MSE_C',
        # 'production_CMLP_DTW_C',
        # 'production_CLSTM_MSE_C',
        # 'production_CLSTM_DTW_C'
    ]
    
    for run in best_runs:
        path = osp.join(production_runs_path, run)
        config = utils_load_config(path)

        # analysis_error_density_plot(config, base_path=path, prefix='best')
        analysis_auto_plot_profiles(config, k=10, base_path=path, prefix='best')

        # Usage example for latex table creation
        headers = p8_names_latex  # Example headers
        row_filter = [3536, 2077, 560]  # List of IDs to include in the table
        csv_to_latex_table(config, row_filter, base_path=path, prefix='best', headers=headers)

    print('\n Completed! \n')
