import os
import os.path as osp
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import rc

matplotlib.use('Agg')
mpl.rc('text', usetex=True)


# -----------------------------------------------------------------
#  global variables
# -----------------------------------------------------------------

# these colours should be colour-blind friendly
colours = ['#01332B', '#D81B60', '#1E88E5', '#FFC107']

lines = ['solid', 'solid', 'solid', 'solid']


# -----------------------------------------------------------------
#  Plot loss functions (average loss for training & validation)
# -----------------------------------------------------------------
def compare_loss_function(lf_data, lf_labels, lf_colours, lf_lines,
                          output_dir='./', profile_type='H', file_type='pdf', lf_type='validation'):

    print('Producing loss function plot:')

    fig, ax = plt.subplots(figsize=(7, 5))
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)

    for i in range(len(lf_data)):
        ax.plot(lf_data[i], lw=1.5, c=lf_colours[i], linestyle=lf_lines[i], label=lf_labels[i])

    ax.set_xlabel(r'$\textrm{Training epoch}$', fontsize=15, labelpad=10)
    ax.set_ylabel(r'$\textrm{Average loss}$', fontsize=15, labelpad=10)

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.set_yscale('log')

    ax.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)

    ax.legend(loc='best', frameon=False)

    path = os.path.join(output_dir, 'loss_functions_comparison_%s_%s.%s' % (lf_type, profile_type, file_type))

    fig.savefig(path)
    print('Saved plot to:\t%s' % path)


# -----------------------------------------------------------------
#  H DTW loss
# -----------------------------------------------------------------

base = './DTW_loss_H'

lf_0_val = np.load(osp.join(base, '0_vanilla/data_products/val_loss_H_2000_epochs.npy'))
lf_0_train = np.load(osp.join(base, '0_vanilla/data_products/train_loss_H_2000_epochs.npy'))

lf_1_val = np.load(osp.join(base, '1_vanilla+bn/data_products/val_loss_H_2000_epochs.npy'))
lf_1_train = np.load(osp.join(base, '1_vanilla+bn/data_products/train_loss_H_2000_epochs.npy'))

lf_2_val = np.load(osp.join(base, '2_vanilla+dropout/data_products/val_loss_H_2000_epochs.npy'))
lf_2_train = np.load(osp.join(base, '2_vanilla+dropout/data_products/train_loss_H_2000_epochs.npy'))

lf_3_val = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/val_loss_H_2000_epochs.npy'))
lf_3_train = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/train_loss_H_2000_epochs.npy'))


lf_data_val = [lf_0_val, lf_1_val, lf_2_val, lf_3_val]

lf_labels_val = ['DTW loss - MLP1', 'DTW loss - MLP1 + BN', 'DTW loss - MLP1 + DO', 'DTW loss - MLP1 + BN + DO']

compare_loss_function(lf_data_val, lf_labels_val, colours, lines, profile_type='H', lf_type='DTW_validation')

lf_data_train = [lf_0_train, lf_1_train, lf_2_train, lf_3_train]

lf_labels_val = ['DTW loss - MLP1', 'DTW loss - MLP1 + BN', 'DTW loss - MLP1 + DO', 'DTW loss - MLP1 + BN + DO']
compare_loss_function(lf_data_train, lf_labels_val, colours, lines, profile_type='H', lf_type='DTW_training')

# -----------------------------------------------------------------
#  T DTW loss
# -----------------------------------------------------------------

base = './DTW_loss_T'

lf_0_val = np.load(osp.join(base, '0_vanilla/data_products/val_loss_T_2000_epochs.npy'))
lf_0_train = np.load(osp.join(base, '0_vanilla/data_products/train_loss_T_2000_epochs.npy'))

lf_1_val = np.load(osp.join(base, '1_vanilla+bn/data_products/val_loss_T_2000_epochs.npy'))
lf_1_train = np.load(osp.join(base, '1_vanilla+bn/data_products/train_loss_T_2000_epochs.npy'))

lf_2_val = np.load(osp.join(base, '2_vanilla+dropout/data_products/val_loss_T_2000_epochs.npy'))
lf_2_train = np.load(osp.join(base, '2_vanilla+dropout/data_products/train_loss_T_2000_epochs.npy'))

lf_3_val = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/val_loss_T_2000_epochs.npy'))
lf_3_train = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/train_loss_T_2000_epochs.npy'))


lf_data_val = [lf_0_val, lf_1_val, lf_2_val, lf_3_val]

lf_labels_val = ['DTW loss - MLP1', 'DTW loss - MLP1 + BN', 'DTW loss - MLP1 + DO', 'DTW loss - MLP1 + BN + DO']

compare_loss_function(lf_data_val, lf_labels_val, colours, lines, profile_type='T', lf_type='DTW_validation')

lf_data_train = [lf_0_train, lf_1_train, lf_2_train, lf_3_train]

lf_labels_val = ['DTW loss - MLP1', 'DTW loss - MLP1 + BN', 'DTW loss - MLP1 + DO', 'DTW loss - MLP1 + BN + DO']
compare_loss_function(lf_data_train, lf_labels_val, colours, lines, profile_type='T', lf_type='DTW_training')

# -----------------------------------------------------------------
#  H MSE loss
# -----------------------------------------------------------------

base = './MSE_loss_H'

lf_0_val = np.load(osp.join(base, '0_vanilla/data_products/val_loss_H_2000_epochs.npy'))
lf_0_train = np.load(osp.join(base, '0_vanilla/data_products/train_loss_H_2000_epochs.npy'))

lf_1_val = np.load(osp.join(base, '1_vanilla+bn/data_products/val_loss_H_2000_epochs.npy'))
lf_1_train = np.load(osp.join(base, '1_vanilla+bn/data_products/train_loss_H_2000_epochs.npy'))

lf_2_val = np.load(osp.join(base, '2_vanilla+dropout/data_products/val_loss_H_2000_epochs.npy'))
lf_2_train = np.load(osp.join(base, '2_vanilla+dropout/data_products/train_loss_H_2000_epochs.npy'))

lf_3_val = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/val_loss_H_2000_epochs.npy'))
lf_3_train = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/train_loss_H_2000_epochs.npy'))


lf_data_val = [lf_0_val, lf_1_val, lf_2_val, lf_3_val]

lf_labels_val = ['MSE loss - MLP1', 'MSE loss - MLP1 + BN', 'MSE loss - MLP1 + DO', 'MSE loss - MLP1 + BN + DO']

compare_loss_function(lf_data_val, lf_labels_val, colours, lines, profile_type='H', lf_type='MSE_validation')

lf_data_train = [lf_0_train, lf_1_train, lf_2_train, lf_3_train]

lf_labels_val = ['MSE loss - MLP1', 'MSE loss - MLP1 + BN', 'MSE loss - MLP1 + DO', 'MSE loss - MLP1 + BN + DO']
compare_loss_function(lf_data_train, lf_labels_val, colours, lines, profile_type='H', lf_type='MSE_training')

# -----------------------------------------------------------------
#  T MSE loss
# -----------------------------------------------------------------

base = './MSE_loss_T'

lf_0_val = np.load(osp.join(base, '0_vanilla/data_products/val_loss_T_2000_epochs.npy'))
lf_0_train = np.load(osp.join(base, '0_vanilla/data_products/train_loss_T_2000_epochs.npy'))

lf_1_val = np.load(osp.join(base, '1_vanilla+bn/data_products/val_loss_T_2000_epochs.npy'))
lf_1_train = np.load(osp.join(base, '1_vanilla+bn/data_products/train_loss_T_2000_epochs.npy'))

lf_2_val = np.load(osp.join(base, '2_vanilla+dropout/data_products/val_loss_T_2000_epochs.npy'))
lf_2_train = np.load(osp.join(base, '2_vanilla+dropout/data_products/train_loss_T_2000_epochs.npy'))

lf_3_val = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/val_loss_T_2000_epochs.npy'))
lf_3_train = np.load(osp.join(base, '3_vanilla+bn+dropout/data_products/train_loss_T_2000_epochs.npy'))


lf_data_val = [lf_0_val, lf_1_val, lf_2_val, lf_3_val]

lf_labels_val = ['MSE loss - MLP1', 'MSE loss - MLP1 + BN', 'MSE loss - MLP1 + DO', 'MSE loss - MLP1 + BN + DO']

compare_loss_function(lf_data_val, lf_labels_val, colours, lines, profile_type='T', lf_type='MSE_validation')

lf_data_train = [lf_0_train, lf_1_train, lf_2_train, lf_3_train]

lf_labels_val = ['MSE loss - MLP1', 'MSE loss - MLP1 + BN', 'MSE loss - MLP1 + DO', 'MSE loss - MLP1 + BN + DO']

compare_loss_function(lf_data_train, lf_labels_val, colours, lines, profile_type='T', lf_type='MSE_training')
