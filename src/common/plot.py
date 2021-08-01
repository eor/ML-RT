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
def plot_profile_single(profile_true, profile_inferred, n_epoch, output_dir, profile_type, prefix ,parameters=None):

    # -----------------------------------------------------------------
    # figure setup
    # -----------------------------------------------------------------
    
    fig = plt.figure(figsize=(10,7))
    gs = fig.add_gridspec(nrows = 2,ncols= 1, hspace=0,height_ratios= [3,1])
    ax_array = gs.subplots(sharex=True, sharey=False)


    rc('font',**{'family':'serif'})
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
            # change line after every 4 parameters are added  
            if j!= 0 and j%5 == 0:
                a += '\n'
            # append the paramter with it's name and value to title string
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
    ax_array[0].set_xticks(np.arange(0, len(profile_true), step=50),minor = True)
    ax_array[0].tick_params(axis='both', which='both',right= True, top = True)
    
    # -----------------------------------------------------------------
    # second plot (diff / relative error)
    # -----------------------------------------------------------------

    # addition of small number to denominator to avoid divsion by zero
    relative_error = (profile_true - profile_inferred)/ (np.fabs(profile_true)+1.0e-6)
    ax_array[1].plot(relative_error, c='black', label='Relative error', linewidth=0.6)
    ax_array[1].grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax_array[1].set_ylabel(r'Rel error', fontsize=14)
    ax_array[1].set_xlabel(r'Radius $\mathrm{[kpc]}$',fontsize=14)
    ax_array[1].set_xticks(np.arange(0, len(profile_true), step=50),minor = True)
    ax_array[1].tick_params(axis='both', which='both',right= True, top = True)

    # -----------------------------------------------------------------
    # get MSE and construct file name
    # -----------------------------------------------------------------
    mse = (np.square(profile_true - profile_inferred)).mean()
    mse = np.log10(mse+1.0e-11)
    fileName = '{:s}_{:s}_profile_epoch_{:d}_logMSE_{:.4e}'.format(profile_type, prefix, n_epoch, mse)
    fileName = fileName.replace('.', '_')
    fileName += '.png'

    plt.savefig(os.path.join(output_dir,fileName))

    # -----------------------------------------------------------------
    # clean up, necessary when making lots of plots
    # -----------------------------------------------------------------
    for ax in ax_array:
        ax.clear()
        plt.close('all')


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there!')