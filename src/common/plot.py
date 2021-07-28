import os
import os.path as osp
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

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
        ax.plot(gen, lw=1.5, c='orange', label='Generator')
        ax.plot(dis, lw=1.5, c='teal', label='Discriminator')
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

    path = os.path.join(output_dir, '%s_loss_functions_%d_epochs_lr_%5f.%s'%(profile_type, epoch, lr, file_type))

    fig.savefig(path)
    print('Saved plot to:\t%s'%(path))


#-----------------------------------------------------------------
# Plot a single profile comparison (ground truth vs inferred)
#-----------------------------------------------------------------
def plot_profile_single(profileTrue, profileInferred, nEpoch, output_dir, profileChoice, parameters=None):

    # TODO: fix this function, change variables to conform with naming scheme (no more camel casing).

    # -----------------------------------------------------------------
    # figure setup
    # -----------------------------------------------------------------
    # fig, (ax1, axArray[1]) = plt.subplots(2,1, figsize=(10,6))
    #fig = plt.figure( figsize=(10,7))

    fig, axArray = plt.subplots(5, 1, sharex=True, figsize=(10,10))

    fig.subplots_adjust(hspace=0)

    #ax1 = fig.add_subplot(3,1,1)
    #ax2 = fig.add_subplot(3,1,2)
    #ax3 = fig.add_subplot(3,1,3)

    #plt.subplots_adjust(wspace=0, hspace=0)
    # plt.subplots_adjust(wspace=0, hspace=-0.4)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)
#
#     # -----------------------------------------------------------------
#     # add parameters as title
#     # -----------------------------------------------------------------
#     pNames = ['$\log_{10}{M_\mathrm{halo}}', '$z', '$t_\mathrm{source}',
#               '$-\\alpha_\mathrm{QSO}','$f_\mathrm{esc}']
#
#
#
#     #  ['$\log_{10}{M_\mathrm{halo}}$', '$z$', '$t_\mathrm{source}$',
#     #  '$-\\alpha_{\mathrm{QSO}}$','$\epsilon_\mathrm{QSO}$','$f_\mathrm{esc,\\ast}$','$\\alpha_\mathrm{\\ast,IMF}$','$m_\mathrm{min,\\ast,IMF}$'
#
#
#
#     if len(parameters)>0:
#         a = ''
#         for j in range(len(pNames)):
#             value = parameters[j]
#             name = pNames[j]
#             a = a + name +' = '+str(value)+'$'
#             if j==2:
#                 a += '$\mathrm{Myr}$'
#
#             a +=  '\, \, \, '
#
#     fig.suptitle(a, fontsize=12)
#
#     # -----------------------------------------------------------------
#     # first plot (true and inferred profiles)
#     # -----------------------------------------------------------------
#     axArray[0].plot(profileTrue, c='green', label='Truth')
#     axArray[0].plot(profileInferred, c='orange', label='Reconstruction')
#
#     #axArray[0].set_xlabel(r'Radius $\mathrm{[kpc]}$',fontsize=14 )
#
#     if profileChoice == 'H':
#         axArray[0].set_ylabel(r'$\log_{10}(x_{H_{II}}) $', fontsize=14)
#     elif profileChoice == 'T':
#         axArray[0].set_ylabel(r'$\log_{10}(T_{\mathrm{kin}}/\mathrm{K}) $', fontsize=14)
#     else:
#         axArray[0].set_ylabel(r'Physical Unit', fontsize=14)
#
#     axArray[0].minorticks_on()
#     axArray[0].legend(loc='upper right', frameon=False)
#     axArray[0].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
#     #axArray[0].axes.xaxis.set_ticklabels([])
#     #axArray[0].tick_params(axis='x', which='both', length=0.)
#     # if profileChoice=='H':
#     #     # axArray[0].set_ylim(-8.0, 0.2)
#     #     #axArray[0].set_ylim(-0.5, 1.5)
#     #     axArray[0].set_ylim(-11, 1)
#     # else:
#     #     axArray[0].set_ylim(0.0,10.0)
#
#     # -----------------------------------------------------------------
#     # second plot (diff / relative error)
#     # -----------------------------------------------------------------
#     relativeError = (profileTrue - profileInferred) / np.fabs(profileTrue)
#
#     axArray[1].plot(relativeError, c='black', label='Relative error', linewidth=0.6)
#     axArray[1].minorticks_on()
#     axArray[1].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
#     #axArray[1].axes.xaxis.set_ticklabels([])
#     #axArray[1].tick_params(axis='x', which='both', length=0.)
#     axArray[1].set_ylabel(r'Rel error', fontsize=14)
#     #axArray[1].set_yscale('log')
#
#
#     # ratio1 = 0.3
#     # axArray[0].set_aspect(1.0/axArray[0].get_data_ratio()*ratio1)
#
#
#     #ratio2 = 0.12
#     #axArray[1].set_aspect(1.0/axArray[1].get_data_ratio()*ratio2)
#
#     # -----------------------------------------------------------------
#     # third plot (absolute error)
#     # -----------------------------------------------------------------
#     absoluteError = (profileTrue - profileInferred)
#
#     axArray[2].plot(absoluteError, c='black', label='Absolute error', linewidth=0.6)
#     axArray[2].minorticks_on()
#     axArray[2].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
#     axArray[2].set_xlabel(r'Radius $\mathrm{[kpc]}$',fontsize=14)
#     axArray[2].set_ylabel(r'Abs error', fontsize=14)
#     #axArray[1].set_yscale('log')
#
#     # ratio1 = 0.25
#     # axArray[0].set_aspect(1.0/axArray[0].get_data_ratio()*ratio1)
#     #ratio3 = 0.12
#     #axArray[2].set_aspect(1.0/axArray[2].get_data_ratio()*ratio2)
#     #axArray[2].set(aspect=1.0/axArray[2].get_data_ratio()*ratio2, adjustable='box')
#
#     # -----------------------------------------------------------------
#     # fourth plot (profile derivative)
#     # -----------------------------------------------------------------
#     derivative = utils_derivative_1(profileTrue, absolute=True, mode='numpy')
#     derivative += 1.0e-2
#     axArray[3].plot(derivative, c='green', label='Absolute error', linewidth=0.6)
#     axArray[3].minorticks_on()
#     axArray[3].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
#     axArray[3].set_xlabel(r'Radius $\mathrm{[kpc]}$',fontsize=14)
#     axArray[3].set_ylabel(r'Derivative', fontsize=14)
#
#
#     # -----------------------------------------------------------------
#     # fourth plot (profile derivative)
#     # -----------------------------------------------------------------
#
#     mySigma = 25
#     # boost = 100
#     # derivative *= boost
#     derivative = gaussian_filter(derivative, sigma=mySigma)
#
#     axArray[4].plot(derivative, c='green', label='Absolute error', linewidth=0.6)
#     axArray[4].minorticks_on()
#     axArray[4].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
#     axArray[4].set_xlabel(r'Radius $\mathrm{[kpc]}$',fontsize=14)
#     axArray[4].set_ylabel(r'Derivative changed', fontsize=14)
#
#
#     # -----------------------------------------------------------------
#     # get MSE and construct file name
#     # -----------------------------------------------------------------
#     mse = (np.square( profileTrue - profileInferred)).mean()
#     mse = np.log10(mse+1.0e-11)
#
#     fileName = '{:s}_profile_epoch_{:d}_logMSE_{:.4e}'.format(profileChoice, nEpoch, mse)
#     fileName = fileName.replace('.', '_')
#     fileName += '.png'
#
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     fig.savefig(os.path.join(output_dir,fileName))
#
#     # -----------------------------------------------------------------
#     # clean up, necessary when making lots of plots
#     # -----------------------------------------------------------------
#     axArray[0].clear()
#     axArray[1].clear()
#     axArray[2].clear()
#     axArray[3].clear()
#     axArray[4].clear()
#     plt.close('all')


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there!')