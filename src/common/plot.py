import os
import os.path as osp
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
try:
    from parameter_settings import p5_names_latex, p8_names_latex
except ImportError:
    from common.parameter_settings import p5_names_latex, p8_names_latex

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
def plot_profile_single(profile_true, profile_inferred, n_epoch, output_dir, profile_type, prefix, parameters=None):

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
    
    if parameters is not None and len(parameters)>0:
        a = ''
        for j in range(len(param_names)):
            # add line break after every 4 parameters are added
            if j!= 0 and j%5 == 0:
                a += '\n'
            # append the parameter with its name and value to title string
            value = parameters[j]
            name = '$' + param_names[j]
            a = a + name +' = '+str(value)+'$'
            if j==2:
                a += '$\mathrm{Myr}$'
            a +=  '\, \, \, '

    fig.suptitle(a, fontsize=12)

    # -----------------------------------------------------------------
    # first plot (true and inferred profiles)
    # -----------------------------------------------------------------
    
    ax_array[0].plot((profile_true), c='green', label='Truth')
    ax_array[0].plot((profile_inferred), c='orange', label='Reconstruction')

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
    relative_error = (profile_true - profile_inferred)/ (np.fabs(profile_true)+1.0e-6)
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
def plot_parameter_space_MSE(parameters, profilesTrue, profilesInfer,
                             profileChoice, nEpoch, outDir='./', prefix='test'):

    # NOTE: This is a work in progress, only checked in forbackup and testing purposes

    print('Making parameter-MSE plot: {} set, {} profiles, {} epochs'.format(prefix, profileChoice, nEpoch))

    # Reminder: we are using the following parameters
    # 1. haloMassLog        interval=[8.0, 15.0]
    # 2. redshift           interval=[6.0, 13.0]
    # 3. sourceAge          interval=[0.1, 20.0]
    # 4. qsoAlpha           interval=[1.0, 2.0]
    # 5. starsEscFrac       interval=[0.0, 1.0]

    # parameter labels
    pLabels = ['$\log(\mathrm{M}_{\mathrm{halo}})$',
               '$z$',
               '$t_{\mathrm{Age}}$',
               '$-\\alpha_{\mathrm{QSO}}$',
               '$f_{\mathrm{esc}}$']

    # parameter intervals (with some padding)
    # limits = [(8.0, 15.0), (6.0, 13.0), (0.1, 20.0), (1.0, 2.0), (0.0, 1.0)]
    limits = [(7.5, 15.5), (5.5, 13.5), (0.5, 21.2), (0.94, 2.07), (-0.075, 1.075)]

    nProfiles = np.shape(parameters)[0]
    nParameters = np.shape(parameters)[1]
    N = nParameters

    # 2. compute MSE
    mseArray = (np.square( (profilesTrue) - (profilesInfer))).mean(axis=1)
    #mseArray = (np.square( 10**(profilesTrue) - 10**(profilesInfer))).mean(axis=1)
    mseArray = np.log10(mseArray + 1e-11)

    # 3. set up main plot
    #f, axarr = plt.subplots(N - 1, N - 1, sharex='col', sharey='row', figsize=(12, 12))
    f, axarr = plt.subplots(N - 1, N - 1, figsize=(12, 12))
    for i in range(0, N - 1):
        for j in range(1, N):
            if j > i:

                ax = axarr[N - 2 - i, j - 1].scatter(x=parameters[:,j], y=parameters[:,i], c=mseArray[:],
                                                     marker='h', s=150, alpha=0.75, edgecolors='none',
                                                     cmap=mpl.cm.inferno_r)

                axarr[N - 2 - i, j - 1].set_ylim(limits[i])
                axarr[N - 2 - i, j - 1].set_xlim(limits[j])

                if i == 0:
                    # bottom row
                    axarr[N - 2 - i, j - 1].tick_params(axis='x', which='major', labelsize=12)
                    axarr[N - 2 - i, j - 1].set_xlabel(xlabel=r'$\textrm{%s}$' % pLabels[j], size=20, labelpad=10)

                    if i != j - 1:
                        # turn of labels and ticks all panels except the leftmost panel
                        axarr[N - 2 - i, j - 1].tick_params(axis='y', which='both', length=0.)
                        axarr[N - 2 - i, j - 1].axes.yaxis.set_ticklabels([])


                if i == j - 1:
                    # leftmost panel in each row
                    axarr[N - 2 - i, j - 1].tick_params(axis='y', which='major', labelsize=12)
                    axarr[N - 2 - i, j - 1].set_ylabel(ylabel=r'$\textrm{%s}$' % pLabels[i], size=20, labelpad=10)

                if (i != 0) and (i != j - 1):
                    # turn of labels & ticks for all panels not in the bottom row and not leftmost
                    axarr[N - 2 - i, j - 1].tick_params(axis='y', which='both', length=0.)
                    axarr[N - 2 - i, j - 1].axes.yaxis.set_ticklabels([])



                axarr[N - 2 - i, j - 1].set_aspect('auto')

            else:
                # do not draw the panels above the diagonal
                axarr[N - 2 - i, j - 1].axis('off')

    # 4. add color bar &  minor adjustments
    cax = f.add_axes((0.07,0.93,0.5,0.03), frameon=True, xticks=[], yticks=[])
    cBar = f.colorbar(ax, cax, orientation='horizontal')
    cBar.set_label(label=r'$\textrm{log}_{10} (\textrm{MSE of true and inferred profiles})$',weight='bold',size=20, labelpad=20)
    cBar.ax.tick_params(labelsize=17)

    # make good use of space
    f.subplots_adjust(hspace=0, wspace=0, left=0.07, bottom=0.07, right=0.95, top=0.98)

    # 5. build file name & save file
    fileName = 'parameter_space_MSE_%s_%d_Epochs.pdf'%(profileChoice, nEpoch)
    if prefix:
        fileName = prefix +'_'+fileName

    outFile = osp.join(outDir, fileName)
    f.savefig(outFile)





# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there!')