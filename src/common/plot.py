import os
import os.path as osp
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
try:
    from parameter_settings import p5_names_latex, p8_names_latex, p5_limits, p8_limits
except ImportError:
    from common.parameter_settings import p5_names_latex, p8_names_latex, p5_limits, p8_limits

try:
    from utils import *
except ImportError:
    from common.utils import *

from scipy.ndimage import gaussian_filter


matplotlib.use('Agg')
mpl.rc('text', usetex=True)


# -----------------------------------------------------------------
#  Plot loss functions (average loss for training & validation)
# -----------------------------------------------------------------
def plot_loss_function(lf1, lf2, epoch, lr, output_dir='./', profile_type='H', file_type='png', gan=False):

    print('Producing loss function plot:')

    fig, ax = plt.subplots(figsize=(7, 5))
    rc('font', **{'family':'serif'})
    rc('text', usetex=True)

    if gan:
        ax.plot(lf1, lw=1.5, c='orange', label='Generator')
        ax.plot(lf2, lw=1.5, c='teal', label='Discriminator')
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
                        profile_type, prefix, parameters=None):

    # TODO: add file_type argument to enable pdf and png outputs

    # -----------------------------------------------------------------
    # figure setup
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(10,7))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3,1])
    ax_array = gs.subplots(sharex=True, sharey=False)

    rc('font', **{'family': 'serif'})
    rc('text', usetex=True) 
    
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
            # add line break after every 4 parameters are added
            if j!= 0 and j%5 == 0:
                a += '\n'
            # append the parameter with its name and value to title string
            value = parameters[j]
            name = '$' + param_names[j]
            a = a + name + ' = ' + str(value) + '$'
            if j==2:
                a += '$\mathrm{Myr}$'
            a += '\, \, \, '

    fig.suptitle(a, fontsize=12)

    # -----------------------------------------------------------------
    # first plot (true and inferred profiles)
    # -----------------------------------------------------------------
    ax_array[0].plot(profile_true, c='green', label='Truth')
    ax_array[0].plot(profile_inferred, c='orange', label='Reconstruction')

    if profile_type == 'H':
        ax_array[0].set_ylabel(r'$\log_{10}(x_{H_{II}}) $', fontsize=14)
    elif profile_type == 'T':
        ax_array[0].set_ylabel(r'$\log_{10}(T_{\mathrm{kin}}/\mathrm{K}) $', fontsize=14)
    else:
        ax_array[0].set_ylabel(r'Physical Unit', fontsize=14)

    ax_array[0].legend(loc='upper right', frameon=False)
    ax_array[0].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax_array[0].set_xticks(np.arange(0, len(profile_true), step=50), minor=True)
    ax_array[0].tick_params(axis='both', which='both', right=True, top=True)
    
    # -----------------------------------------------------------------
    # second plot (diff / relative error)
    # -----------------------------------------------------------------
    # addition of small number to denominator to avoid division by zero
    relative_error = (profile_true - profile_inferred) / (np.fabs(profile_true)+1.0e-6)
    ax_array[1].plot(relative_error, c='black', label='Relative error', linewidth=0.6)
    ax_array[1].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax_array[1].set_ylabel(r'Rel error', fontsize=14)
    ax_array[1].set_xlabel(r'Radius $\mathrm{[kpc]}$',fontsize=14)
    ax_array[1].set_xticks(np.arange(0, len(profile_true), step=50), minor=True)
    ax_array[1].tick_params(axis='both', which='both', right=True, top=True)

    # -----------------------------------------------------------------
    # get MSE and construct file name
    # -----------------------------------------------------------------
    mse = (np.square(profile_true - profile_inferred)).mean()
    mse = np.log10(mse+1.0e-11)
    file_name = '{:s}_{:s}_profile_epoch_{:d}_logMSE_{:.4e}'.format(profile_type, prefix, n_epoch, mse)
    file_name = file_name.replace('.', '_')
    file_name += '.png'

    plt.savefig(os.path.join(output_dir, file_name))

    # -----------------------------------------------------------------
    # clean up, necessary when making lots of plots
    # -----------------------------------------------------------------
    for ax in ax_array:
        ax.clear()
    plt.close('all')


# -----------------------------------------------------------------
#  visualise MSE (truth vs inference) for whole parameter space
# -----------------------------------------------------------------
def plot_parameter_space_mse(parameters, profiles_true, profiles_gen, profile_type,
                             n_epoch, output_dir='./', prefix='test'):

    print('Making parameter-MSE plot: {} set, {} profiles, {} epochs'.format(prefix, profile_type, n_epoch))

    # -----------------------------------------------------------------
    # get parameters labels
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
                 (6.0 - 0.25, 13.0 + 0.2),
                 (0.1 + 0.2, 21.0 + 0.1),
                 (0.0 - 0.09, 2.0 + 0.09),
                 (0.0 - 0.09, 1.0 + 0.09),
                 (0.0 - 0.09, 1.0 + 0.09),
                 (0.0 - 0.09, 2.0 + 0.09),
                 (0.6989700043360189 - 0.09, 2.6989700043360187 + 0.09)]


    if N == 5:
        padding = padding_5
    else:
        padding = padding_8






    # TODO: save as png or pdf
    # TODO: ability to fix color range (to make different plots comparable)



    # 2. compute MSE
    mse_array = (np.square( (profiles_true) - (profiles_gen))).mean(axis=1)
    #mse_array = (np.square( 10**(profiles_true) - 10**(profiles_gen))).mean(axis=1)
    mse_array = np.log10(mse_array + 1e-11)

    # 3. set up main plot
    # marker size
    if N == 5:
        marker_size = 150
    else:
        marker_size = 15

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
                    ax_array[N - 2 - i, j - 1].tick_params(axis='x', which='major', labelsize=12)
                    ax_array[N - 2 - i, j - 1].set_xlabel(xlabel=r'$\textrm{%s}$' % p_labels[j], size=20, labelpad=10)

                    if i != j - 1:
                        # turn of labels and ticks all panels except the leftmost panel
                        ax_array[N - 2 - i, j - 1].tick_params(axis='y', which='both', length=0.)
                        ax_array[N - 2 - i, j - 1].axes.yaxis.set_ticklabels([])

                if i == j - 1:
                    # leftmost panel in each row
                    ax_array[N - 2 - i, j - 1].tick_params(axis='y', which='major', labelsize=12)
                    ax_array[N - 2 - i, j - 1].set_ylabel(ylabel=r'$\textrm{%s}$' % p_labels[i], size=20, labelpad=10)

                if (i != 0) and (i != j - 1):
                    # turn of labels & ticks for all panels not in the bottom row and not leftmost
                    ax_array[N - 2 - i, j - 1].tick_params(axis='y', which='both', length=0.)
                    ax_array[N - 2 - i, j - 1].axes.yaxis.set_ticklabels([])

                ax_array[N - 2 - i, j - 1].set_aspect('auto')

            else:
                # do not draw the panels above the diagonal
                ax_array[N - 2 - i, j - 1].axis('off')

    # 4. add color bar &  minor adjustments
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

    # 5. build file name & save file
    file_name = 'parameter_space_MSE_%s_%d_epochs.pdf'%(profile_type, n_epoch)
    if prefix:
        file_name = prefix + '_' + file_name

    output_file = osp.join(output_dir, file_name)
    f.savefig(output_file)


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there!')







