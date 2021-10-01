import os
import os.path as osp
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns

try:
    from settings_parameters import p5_names_latex, p8_names_latex, p5_limits, p8_limits
except ImportError:
    from common.settings_parameters import p5_names_latex, p8_names_latex, p5_limits, p8_limits

try:
    from utils import utils_compute_mse, utils_compute_dtw, utils_get_current_timestamp
except ImportError:
    from common.utils import utils_compute_mse, utils_compute_dtw, utils_get_current_timestamp

from scipy.ndimage import gaussian_filter


matplotlib.use('Agg')
mpl.rc('text', usetex=True)


# -----------------------------------------------------------------
#  Plot loss functions (average loss for training & validation)
# -----------------------------------------------------------------
def plot_loss_function(lf1, lf2, lf3, epoch, lr, output_dir='./', profile_type='H', file_type='png', gan=False):

    print('Producing loss function plot:')

    fig, ax = plt.subplots(figsize=(7, 5))
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    if gan:
        ax.plot(lf1, lw=1.5, c='orange', label='Generator')
        ax.plot(lf2, lw=1.5, c='teal', label='Discriminator\_real')
        ax.plot(lf3, lw=1.5, c='red', label='Discriminator\_fake')
    else:
        ax.plot(lf1, lw=1.5, c='red', label='Training')
        ax.plot(lf2, lw=1.5, c='blue', label='Validation')

    ax.set_xlabel(r'$\textrm{Training epoch}$', fontsize=15, labelpad=10)
    ax.set_ylabel(r'$\textrm{Average loss}$', fontsize=15, labelpad=10)

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.set_yscale('log')

    ax.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    # ax.grid(which='minor', color='#cccccc', linestyle='-', linewidth='0.4', alpha=0.4)

    # ax.legend(loc='upper right', frameon=False)
    ax.legend(loc='best', frameon=False)

    path = os.path.join(output_dir, '%s_loss_functions_%d_epochs_lr_%5f.%s' % (profile_type, epoch, lr, file_type))

    fig.savefig(path)
    print('Saved plot to:\t%s' % path)


# -----------------------------------------------------------------
# Plot a single profile comparison (ground truth vs inferred)
# -----------------------------------------------------------------
def plot_profile_single(profile_true, profile_inferred, n_epoch, output_dir,
                        profile_type, prefix, profile_order=['H', 'T', 'He_II', 'He_III'],
                        file_type='pdf', parameters=None, add_errors=True):

    def get_label_Y(profile_type):
        if profile_type == 'H':
            return r'$\log_{10}(x_{H_{II}}) $'
        elif profile_type == 'T':
            return r'$\log_{10}(T_{\mathrm{kin}}/\mathrm{K}) $'
        elif profile_type == 'He_II':
            return r'$\log_{10}(x_{He_{II}}) $'
        elif profile_type == 'He_III':
            return r'$\log_{10}(x_{He_{III}}) $'
        else:
            return r'Physical Unit'

    # convert inputs to usable forms
    if len(profile_true.shape) != 2:
        profile_true = profile_true[np.newaxis, :]
    if len(profile_inferred.shape) != 2:
        profile_inferred = profile_inferred[np.newaxis, :]

    num_plots = profile_true.shape[0]

    # -----------------------------------------------------------------
    # figure setup
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(11, 10))
    plt.subplots_adjust(top=0.82)
    
    # compute size of grid ie. rows and columns to fit all the plots
    rows = int(np.sqrt(num_plots))
    columns = int(np.ceil(num_plots / rows))
    
    # outer grid for the plots
    outer = gridspec.GridSpec(rows, columns, wspace=0.3, hspace=0.3)

    # -----------------------------------------------------------------
    # font settings
    # -----------------------------------------------------------------
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    if profile_type == 'C':
        font_size_title = 26
        font_size_ticks = 16
        font_size_legends = 12
        font_size_x_y = 18
    else:
        font_size_title = 26
        font_size_ticks = 26
        font_size_legends = 22
        font_size_x_y = 30

    for i in range(num_plots):

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.0, height_ratios=[3, 1])

        ax0 = fig.add_subplot(inner[0])
        ax1 = fig.add_subplot(inner[1], sharex=ax0)

        # -----------------------------------------------------------------
        # first plot (true and inferred profiles)
        # -----------------------------------------------------------------
        if np.max(profile_inferred[i]) < 1 and np.abs(np.min(profile_inferred[i])) < 1:
            ax0.set_ylim(-5, 5)

        ax0.plot(profile_true[i], c='green', label='Truth')
        ax0.plot(profile_inferred[i], c='orange', label='Inference')

        mse = utils_compute_mse(profile_true[i], profile_inferred[i])
        dtw = utils_compute_dtw(profile_true[i], profile_inferred[i])

        if add_errors:
            ax0.plot([], [], alpha=.0, label='MSE: %e \n DTW: %e' % (mse, dtw))

        ax0.legend(loc=0, frameon=False, prop={'size': font_size_legends})

        # if profile_type is set to combined, get Y_label using profile_order,
        # else, use profile_type directly.
        if profile_type == 'C':
            ax0.set_ylabel(get_label_Y(profile_order[i]), fontsize=font_size_x_y)
        else:
            ax0.set_ylabel(get_label_Y(profile_type), fontsize=font_size_x_y)
        ax0.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
        ax0.tick_params(axis='y', which='both', right=True, top=True, labelsize=font_size_ticks)
        fig.add_subplot(ax0)

        # -----------------------------------------------------------------
        # second plot (diff / relative error)
        # -----------------------------------------------------------------
        absolute_error = profile_true[i] - profile_inferred[i]
        ax1.plot(absolute_error, c='black', label='Absolute error', linewidth=0.6)
        ax1.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
        ax1.set_ylabel(r'Abs error', fontsize=font_size_x_y)
        ax1.set_xlabel(r'Radius $\mathrm{[kpc]}$', fontsize=font_size_x_y)
        ax1.set_xticks(np.arange(0, len(profile_true), step=50), minor=True)
        ax1.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
    # -----------------------------------------------------------------
    # add parameters as title
    # -----------------------------------------------------------------
    if len(parameters) == 5:
        param_names = p5_names_latex
    else:
        param_names = p8_names_latex

    if parameters is not None and len(parameters) > 0:
        a = ''
        for j in range(len(param_names)):
            # add line break after every 3 parameters are added
            if j != 0 and j % 3 == 0:
                a += '\n'
            # append the parameter with its name and value to title string
            value = parameters[j]
            name = '$' + param_names[j]
            a = a + name + ' = ' + str(value) + '$'
            if j == 2:
                a += '$\mathrm{Myr}$'
            a += '\, \, \, '

    fig.suptitle(a, fontsize=font_size_title, y=0.98)

    #         plt.figtext(0.5, 0.01, 'Errors: MSE: %e DTW: %e' % (mse, dtw), fontsize=font_size_title, ha='center', bbox={"alpha": 0, "pad": 10})

    mse = utils_compute_mse(profile_true, profile_inferred)

    # -----------------------------------------------------------------
    # get MSE and construct file name
    # -----------------------------------------------------------------
    log_mse = np.log10(mse + 1.0e-11)
    file_name = '{:s}_{:s}_profile_epoch_{:d}_logMSE_{:.4e}'.format(profile_type, prefix, n_epoch, log_mse)
    file_name = file_name.replace('.', '_')
    file_name += '.' + file_type

    plt.savefig(os.path.join(output_dir, file_name))
    plt.close('all')

    
# -----------------------------------------------------------------
#  Plot Inference profiles generated using inference.py
# -----------------------------------------------------------------
def plot_inference_profiles(profiles, profile_type, parameters, output_dir='./',
                        labels=['Simulation', 'MLP', 'CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'],
                        file_type='pdf', prefix=None):
    
    # default font size for the plot    
    font_size_title = 26
    font_size_ticks = 26
    font_size_legends = 22
    font_size_x_y = 30
    colors = {'Simulation':'#000000',
              'MLP':'#D81B60',
              'CVAE':'#0072B2',
              'CGAN':'#F0E442',
              'LSTM':'#19D4A1',
              'CMLP':'#E69F00',
              'CLSTM':'#64D500'}


    fig, ax = plt.subplots(figsize=(11, 10))
    plt.subplots_adjust(top=0.82)
    if np.max(profiles) < 1 and np.abs(np.min(profiles)) < 1:
        ax.set_ylim(-5, 5)

    def get_label_Y(profile_type):
        if profile_type == 'H':
            return r'$\log_{10}(x_{H_{II}}) $'
        elif profile_type == 'T':
            return r'$\log_{10}(T_{\mathrm{kin}}/\mathrm{K}) $'
        elif profile_type == 'He_II':
            return r'$\log_{10}(x_{He_{II}}) $'
        elif profile_type == 'He_III':
            return r'$\log_{10}(x_{He_{III}}) $'
        else:
            return r'Physical Unit'

    # convert inputs to usable forms
    if len(profiles.shape) != 2:
        profiles = profiles[np.newaxis, :]

    print('Producing Inference profile plot:')
    
    if len(parameters) == 5:
        param_names = p5_names_latex
    else:
        param_names = p8_names_latex

    if parameters is not None and len(parameters) > 0:
        a = ''
        for j in range(len(param_names)):
            # add line break after every 3 parameters are added
            if j != 0 and j % 3 == 0:
                a += '\n'
            # append the parameter with its name and value to title string
            value = parameters[j]
            name = '$' + param_names[j]
            a = a + name + ' = ' + str(value) + '$'
            if j == 2:
                a += '$\mathrm{Myr}$'
            a += '\, \, \, '

    
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    fig.suptitle(a, fontsize=font_size_title, y=0.98)

    ax.set_xlabel(r'Radius $\mathrm{[kpc]}$', fontsize=font_size_x_y, labelpad=10)
    ax.set_ylabel(get_label_Y(profile_type), fontsize=font_size_x_y, labelpad=10)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)

    ax.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    for i in range(len(labels)):
        if labels[i] in colors.keys():
            ax.plot(profiles[i], c=colors[labels[i]], label=labels[i])
        else:
            color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            ax.plot(profiles[i], c=color, label=labels[i])
            
    ax.legend(loc='best', frameon=False, prop={'size': font_size_legends})
    if prefix is None:
        timestamp = utils_get_current_timestamp()
        path = os.path.join(output_dir, '%s_inference_profile_%s.%s' % (profile_type, timestamp, file_type))
    else:
        path = os.path.join(output_dir, '%s_inference_profile_%s.%s' % (profile_type, prefix, file_type))

    fig.savefig(path)
    print('Saved plot to:\t%s' % path)

# -----------------------------------------------------------------
#  Plot Inference profiles generated using inference.py
# -----------------------------------------------------------------
def plot_inference_time_evolution(profiles, profile_type, parameters, output_dir='./',
                        labels=['Simulation', 'MLP', 'CVAE', 'CGAN', 'LSTM', 'CMLP', 'CLSTM'],
                        file_type='pdf', prefix=None):
    def get_label_Y(profile_type):
        if profile_type == 'H':
            return r'$\log_{10}(x_{H_{II}}) $'
        elif profile_type == 'T':
            return r'$\log_{10}(T_{\mathrm{kin}}/\mathrm{K}) $'
        elif profile_type == 'He_II':
            return r'$\log_{10}(x_{He_{II}}) $'
        elif profile_type == 'He_III':
            return r'$\log_{10}(x_{He_{III}}) $'
        else:
            return r'Physical Unit'

    # default font size for the plot    
    font_size_title = 26
    font_size_ticks = 26
    font_size_legends = 22
    font_size_x_y = 32
    colors = {'Simulation':'#000000',
              'MLP':'#D81B60',
              'CVAE':'#0072B2',
              'CGAN':'#F0E442',
              'LSTM':'#19D4A1',
              'CMLP':'#E69F00',
              'CLSTM':'#64D500'}
    
    isSimulationKnown = 'Simulation' in labels
    if isSimulationKnown:
        start = 1
        num_plots = len(labels) - 1
    else:
        start = 0
        num_plots = len(labels)
    
    # compute size of grid ie. rows and columns to fit all the plots
    columns = 2
    rows = int(np.ceil(num_plots / columns))
            
    # -----------------------------------------------------------------
    # figure setup
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(11, 12))
    plt.subplots_adjust(top=0.86)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    print('Producing Inference profile plot:')

    if np.max(profiles) < 1 and np.abs(np.min(profiles)) < 1:
        ax.set_ylim(-5, 5)

    # convert inputs to usable forms
    if len(profiles.shape) != 2:
        profiles = profiles[np.newaxis, :]

    
    if len(parameters) == 5:
        param_names = p5_names_latex
    else:
        param_names = p8_names_latex

    if parameters is not None and len(parameters) > 0:
        a = ''
        for j in range(len(param_names)):
            # add line break after every 3 parameters are added
            if j != 0 and j % 3 == 0:
                a += '\n'
            # append the parameter with its name and value to title string
            value = parameters[j]
            name = '$' + param_names[j]
            a = a + name + ' = ' + str(value) + '$'
            if j == 2:
                a += '$\mathrm{Myr}$'
            a += '\, \, \, '

    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    fig.suptitle(a, fontsize=font_size_title, y=0.98)
    
    for r, row in enumerate(ax):
        for c, ax0 in enumerate(row):
        
            if labels[start] in colors.keys():
                ax0.plot(profiles[start], c=colors[labels[start]], label=labels[start])
            else:
                color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                ax0.plot(profiles[start], c=colors[labels[start]], label=labels[start])

            if isSimulationKnown:
                ax0.plot(profiles[0], c=colors[labels[0]], label=labels[0])
            
            start += 1
            ax0.minorticks_on()
            ax0.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
            ax0.legend(loc='best', frameon=False, prop={'size': font_size_legends})
            ax0.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)

    fig.text(0.5, 0.04, r'Radius $\mathrm{[kpc]}$', ha='center', fontsize=font_size_x_y)
    fig.text(0.02, 0.5, get_label_Y(profile_type), va='center', rotation='vertical', fontsize=font_size_x_y)
#     ax.set_xlabel(r'Radius $\mathrm{[kpc]}$', fontsize=font_size_x_y, labelpad=10)
#     ax.set_ylabel(get_label_Y(profile_type), fontsize=font_size_x_y, labelpad=10)
            
    if prefix is None:
        timestamp = utils_get_current_timestamp()
        path = os.path.join(output_dir, '%s_inference_profile_%s.%s' % (profile_type, timestamp, file_type))
    else:
        path = os.path.join(output_dir, '%s_inference_profile_%s.%s' % (profile_type, prefix, file_type))

    fig.savefig(path)
    print('Saved plot to:\t%s' % path)


# -----------------------------------------------------------------
#  visualise errors (truth vs inference) for whole parameter space
# -----------------------------------------------------------------
def plot_parameter_space_mse(parameters, profiles_true, profiles_gen, profile_type,
                             n_epoch, output_dir='./', prefix='test', file_type='png'):

    print('Making parameter-MSE plot: {} set, {} profiles, {} epochs'.format(prefix, profile_type, n_epoch))

    # -----------------------------------------------------------------
    # set up parameters labels
    # -----------------------------------------------------------------
    n_profiles = np.shape(parameters)[0]
    n_parameters = np.shape(parameters)[1]
    N = n_parameters

    if N == 5:
        p_labels = p5_names_latex
    else:
        p_labels = p8_names_latex

    for i, label in enumerate(p_labels):
        p_labels[i] = '$' + label + '$'

    # -----------------------------------------------------------------
    # add padding to limits
    # -----------------------------------------------------------------
    # parameter intervals (with some padding)
    # limits = [(8.0, 15.0), (6.0, 13.0), (0.1, 20.0), (1.0, 2.0), (0.0, 1.0)]
    padding_5 = [(7.5, 15.5), (5.5, 13.5), (0.5, 21.2), (0.94, 2.07), (-0.075, 1.075)]

    # padding_8 = [(8.0, 15.0),
    #              (6.0, 13.0),
    #              (0.1, 20.0),
    #              (0.0, 2.0),
    #              (0.0, 1.0),
    #              (0.0, 1.0),
    #              (0.0, 2.5),
    #              (0.6989700043360189, 2.6989700043360187)]

    padding_8 = [(8.0 - 0.2, 15.0 + 0.2),
                 (6.0 - 0.39, 13.0 + 0.2),
                 (0.1 + 0.2, 21.0 + 0.1),
                 (0.0 - 0.14, 2.0 + 0.09),
                 (0.0 - 0.09, 1.0 + 0.09),
                 (0.0 - 0.09, 1.0 + 0.09),
                 (0.0 - 0.14, 2.0 + 0.09),
                 (0.6989700043360189 - 0.09, 2.6989700043360187 + 0.09)]

    if N == 5:
        padding = padding_5
    else:
        padding = padding_8

    # TODO: ability to fix color range (to make different plots comparable)
    # -----------------------------------------------------------------
    #  compute MSE for each sample
    # -----------------------------------------------------------------
    if profile_type == 'C':
        mse_array = (np.square((profiles_true) - (profiles_gen))).mean(axis=(2, 1))
    else:
        mse_array = (np.square((profiles_true) - (profiles_gen))).mean(axis=1)

    # mse_array = (np.square( 10**(profiles_true) - 10**(profiles_gen))).mean(axis=1)
    mse_array = np.log10(mse_array + 1e-11)

    # -----------------------------------------------------------------
    # set marker size
    # -----------------------------------------------------------------
    if N == 5:
        marker_size = 150
    else:
        marker_size = 15

    tick_label_size = 11
    label_size = 18

    # -----------------------------------------------------------------
    # set up main plot
    # -----------------------------------------------------------------
    f, ax_array = plt.subplots(N - 1, N - 1, figsize=(12, 12))
    for i in range(0, N - 1):
        for j in range(1, N):
            if j > i:

                ax = ax_array[N - 2 - i, j - 1].scatter(x=parameters[:, j],
                                                        y=parameters[:, i],
                                                        c=mse_array[:],
                                                        marker='h',
                                                        s=marker_size,
                                                        alpha=0.75,
                                                        edgecolors='none',
                                                        cmap=mpl.cm.inferno_r
                                                        )

                ax_array[N - 2 - i, j - 1].set_ylim(padding[i])
                ax_array[N - 2 - i, j - 1].set_xlim(padding[j])

                if i == 0:
                    # bottom row
                    ax_array[N - 2 - i, j - 1].tick_params(axis='x', which='major', labelsize=tick_label_size)
                    ax_array[N - 2 - i, j - 1].set_xlabel(xlabel=r'$\textrm{%s}$' % p_labels[j], size=label_size, labelpad=10)

                    if i != j - 1:
                        # turn of labels and ticks all panels except the leftmost panel
                        ax_array[N - 2 - i, j - 1].tick_params(axis='y', which='both', length=0.)
                        ax_array[N - 2 - i, j - 1].axes.yaxis.set_ticklabels([])

                if i == j - 1:
                    # leftmost panel in each row
                    ax_array[N - 2 - i, j - 1].tick_params(axis='y', which='major', labelsize=tick_label_size)
                    ax_array[N - 2 - i, j - 1].set_ylabel(ylabel=r'$\textrm{%s}$' % p_labels[i], size=label_size, labelpad=10)

                if (i != 0) and (i != j - 1):
                    # turn of labels & ticks for all panels not in the bottom row and not leftmost
                    ax_array[N - 2 - i, j - 1].tick_params(axis='y', which='both', length=0.)
                    ax_array[N - 2 - i, j - 1].axes.yaxis.set_ticklabels([])

                ax_array[N - 2 - i, j - 1].set_aspect('auto')

            else:
                # do not draw the panels above the diagonal
                ax_array[N - 2 - i, j - 1].axis('off')

    # -----------------------------------------------------------------
    # add color bar &  minor adjustments
    # -----------------------------------------------------------------
    cax = f.add_axes((0.07, 0.93, 0.5, 0.03), frameon=True, xticks=[], yticks=[])
    c_bar = f.colorbar(ax, cax, orientation='horizontal')
    c_bar.set_label(
        label=r'$\textrm{log}_{10} (\textrm{MSE of true and inferred profiles})$',
        weight='bold',
        size=20,
        labelpad=20
    )
    c_bar.ax.tick_params(labelsize=17)

    # make good use of space
    f.subplots_adjust(hspace=0, wspace=0, left=0.07, bottom=0.07, right=0.95, top=0.98)

    # -----------------------------------------------------------------
    # build file name & save file
    # -----------------------------------------------------------------
    if prefix:
        file_name = '%s_%s_parameter_space_MSE_%d_epochs.%s' % (profile_type, prefix, n_epoch, file_type)
    else:
        file_name = '%s_parameter_space_MSE_%d_epochs.%s' % (profile_type, n_epoch, file_type)

    output_file = osp.join(output_dir, file_name)
    f.savefig(output_file)

    print('\tSaved plot to: {} '.format(output_file))


# -----------------------------------------------------------------
#  visualise error distribution: histogram and density plot
# -----------------------------------------------------------------
def plot_error_density_mse(profiles_true, profiles_gen,
                           n_epoch, config,output_dir='./', profile_order=['H','T','He_II','He_III'], 
                           prefix='test', file_type='pdf', add_title=True):

    print('Making frequency density plot: {} set, {} profiles, {} epochs'.format(prefix, config.profile_type, n_epoch))

    profile_type_to_color = {
        'H': 'steelblue',
        'T': 'darkorange',
        'He_II': 'darkgreen',
        'He_III': 'darkgreen'
    }
    # -----------------------------------------------------------------
    #  compute MSE for each sample
    # -----------------------------------------------------------------
    if len(np.shape(profiles_true)) == 2:
        profiles_true = profiles_true[:, np.newaxis, :]
        profiles_gen = profiles_gen[:, np.newaxis, :]
    
    mse_array = (np.square((profiles_true) - (profiles_gen))).mean(axis=2)
    mse_array = np.log10(mse_array + 1e-11)

    for i in range(np.shape(profiles_true)[1]):

        # -----------------------------------------------------------------
        #  set up figure
        # -----------------------------------------------------------------
        f, ax = plt.subplots(figsize=(6, 6))
        rc('font', **{'family': 'serif'})
        rc('text', usetex=True)
        
        # select colour based on profile_type
        if config.profile_type == 'C':
            hist_color = profile_type_to_color[profile_order[i]]
        else:
            hist_color = profile_type_to_color[config.profile_type]

        sns.histplot(mse_array[:, i], kde=True, bins=50, color=hist_color, alpha=0.25, ax=ax, stat='density',
                 edgecolor=None, legend=False)

        ax.set_xlabel(r'$\textrm{log}_{10} (\textrm{MSE of true and inferred profiles})$', fontsize=25, labelpad=10)
        ax.set_ylabel(r'$\textrm{Frequency density}$', fontsize=25, labelpad=10)

        ax.tick_params(axis='y', which='major', labelsize=15)
        ax.tick_params(axis='x', which='major', labelsize=15)

        ax.minorticks_on()
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

        ax.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.25)

        f.subplots_adjust(bottom=0.2, left=0.15)

        # ax.legend(loc='best', frameon=False)
        
        
        if config.profile_type == 'C':
            profile_prefix = profile_order[i]
        else:
            profile_prefix = config.profile_type

        if add_title:
            title_string = "%s -- %s -- %s" % (config.model, config.loss_type, profile_prefix.replace('_','\_'))       
            f.suptitle(title_string, fontsize=25, y=0.95)

        path = os.path.join(output_dir, '%s_frequency_density_mse_%d_epochs.%s' % (profile_prefix, n_epoch, file_type))

        f.savefig(path)
        print('Saved plot to:\t%s' % path)
    plt.close('all')


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there!')
