"""
This is a one-off script to plot the results of the inference test of the
fully trained MLP 1 models (T and x_H)

"""

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

# default font size for the plot
font_size_title = 26
font_size_ticks = 26
font_size_legends = 22
font_size_x_y = 30

# labels
y_label_H = r'$\log_{10}(x_{H_{II}}) $'
y_label_T = r'$\log_{10}(T_{\mathrm{kin}}/\mathrm{K}) $'
x_label = r'Radius $\mathrm{[kpc]}$'


def mlp1_compare_inference_examples(base_dir='./'):

    """
    Loads inferred and simulated data, runs the plotting routines.

    Returns: none
    """

    # 1. load inference data
    inference_123_H = np.load(osp.join(base_dir, 'inference_123_H.npy'))  # shape is (3, 1500)
    inference_123_T = np.load(osp.join(base_dir, 'inference_123_T.npy'))  # shape is (3, 1500)

    inference_4_H = np.load(osp.join(base_dir, 'inference_4_H.npy'))      # shape is (5, 1500)
    inference_4_T = np.load(osp.join(base_dir, 'inference_4_T.npy'))      # shape is (5, 1500)

    # 2. load simulated data
    sim_1_H = np.load(osp.join(base_dir, 'run_1_profile_HII.npy'))
    sim_2_H = np.load(osp.join(base_dir, 'run_2_profile_HII.npy'))
    sim_3_H = np.load(osp.join(base_dir, 'run_3_profile_HII.npy'))

    sim_123_H = np.stack(arrays=[sim_1_H, sim_2_H, sim_3_H], axis=0)

    sim_1_T = np.load(osp.join(base_dir, 'run_1_profile_T.npy'))
    sim_2_T = np.load(osp.join(base_dir, 'run_2_profile_T.npy'))
    sim_3_T = np.load(osp.join(base_dir, 'run_3_profile_T.npy'))

    sim_123_T = np.stack(arrays=[sim_1_T, sim_2_T, sim_3_T], axis=0)

    sim_4_H_t04 = np.load(osp.join(base_dir, 'run_4_t4_profile_HII.npy'))
    sim_4_H_t08 = np.load(osp.join(base_dir, 'run_4_t8_profile_HII.npy'))
    sim_4_H_t12 = np.load(osp.join(base_dir, 'run_4_t12_profile_HII.npy'))
    sim_4_H_t16 = np.load(osp.join(base_dir, 'run_4_t16_profile_HII.npy'))
    sim_4_H_t20 = np.load(osp.join(base_dir, 'run_4_t20_profile_HII.npy'))

    sim_4_H = np.stack(arrays=[sim_4_H_t04, sim_4_H_t08, sim_4_H_t12, sim_4_H_t16, sim_4_H_t20], axis=0)

    sim_4_T_t04 = np.load(osp.join(base_dir, 'run_4_t4_profile_T.npy'))
    sim_4_T_t08 = np.load(osp.join(base_dir, 'run_4_t8_profile_T.npy'))
    sim_4_T_t12 = np.load(osp.join(base_dir, 'run_4_t12_profile_T.npy'))
    sim_4_T_t16 = np.load(osp.join(base_dir, 'run_4_t16_profile_T.npy'))
    sim_4_T_t20 = np.load(osp.join(base_dir, 'run_4_t20_profile_T.npy'))

    sim_4_T = np.stack(arrays=[sim_4_T_t04, sim_4_T_t08, sim_4_T_t12, sim_4_T_t16, sim_4_T_t20], axis=0)

    mlp1_compare_test_123_plot(inf_H=inference_123_H,
                               sim_H=sim_123_H,
                               inf_T=inference_123_T,
                               sim_T=sim_123_T)

    mlp1_compare_test_4_plot(inf_H=inference_4_H,
                             sim_H=sim_4_H,
                             inf_T=inference_4_T,
                             sim_T=sim_4_T)


def mlp1_compare_test_123_plot(inf_H, sim_H, inf_T, sim_T):

    # figure setup
    rc('font', family='serif')
    rc('text', usetex=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 16))
    plt.subplots_adjust(top=0.86)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    # set up plots
    ax1.set_ylabel(y_label_H, fontsize=font_size_x_y, labelpad=10)
    ax1.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
    ax1.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax1.minorticks_on()

    ax2.set_xlabel(x_label, fontsize=font_size_x_y, labelpad=10)
    ax2.set_ylabel(y_label_T, fontsize=font_size_x_y, labelpad=10)
    ax2.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
    ax2.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax2.minorticks_on()

    # set limits
    ax1.set_ylim(-7, 0.2)
    ax1.set_xlim(-15, 1200)
    ax2.set_xlim(-15, 1200)
    # ax2.set_ylim(1, 7)

    epsilon = 1e-23

    # top plot
    ax1.plot(np.log10(sim_H[0] + epsilon), c="green", label=r'Test 1 -- Ground truth')
    ax1.plot(np.log10(sim_H[1] + epsilon), c="green", label=r'Test 2 -- Ground truth')
    ax1.plot(np.log10(sim_H[2] + epsilon), c="green", label=r'Test 3 -- Ground truth')

    ax1.plot(inf_H[0], c="orange", label=r'Test 1 -- Inference')
    ax1.plot(inf_H[1], c="orange", label=r'Test 2 -- Inference')
    ax1.plot(inf_H[2], c="orange", label=r'Test 3 -- Inference')

    # bottom plot
    ax2.plot(np.log10(sim_T[0]), c="green", label=r'Ground truth')
    ax2.plot(np.log10(sim_T[1]), c="green", label=r'_Ground truth')
    ax2.plot(np.log10(sim_T[2]), c="green", label=r'_Ground truth')

    ax2.plot(inf_T[0], c="orange", label=r'Inference')
    ax2.plot(inf_T[1], c="orange", label=r'_t=4')
    ax2.plot(inf_T[2], c="orange", label=r'_t=4')

    # legends
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=font_size_legends)

    # annotations
    ax1.text(x=0.065, y=0.30, s=r'\textbf{(2)}', transform=ax1.transAxes, fontsize=font_size_legends)
    ax1.text(x=0.130, y=0.58, s=r'\textbf{(1)}', transform=ax1.transAxes, fontsize=font_size_legends)
    ax1.text(x=0.760, y=0.87, s=r'\textbf{(3)}', transform=ax1.transAxes, fontsize=font_size_legends)

    ax2.text(x=0.035, y=0.06, s=r'\textbf{(2)}', transform=ax2.transAxes, fontsize=font_size_legends)
    ax2.text(x=0.230, y=0.25, s=r'\textbf{(1)}', transform=ax2.transAxes, fontsize=font_size_legends)
    ax2.text(x=0.530, y=0.54, s=r'\textbf{(3)}', transform=ax2.transAxes, fontsize=font_size_legends)

    # save fig
    path = 'test_run_123.pdf'
    fig.savefig(path)


def mlp1_compare_test_4_plot(inf_H, sim_H, inf_T, sim_T):

    # figure setup
    rc('font', family='serif')
    rc('text', usetex=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 16))
    plt.subplots_adjust(top=0.86)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    # set up plots
    ax1.set_ylabel(y_label_H, fontsize=font_size_x_y, labelpad=10)
    ax1.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
    ax1.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax1.minorticks_on()

    ax2.set_xlabel(x_label, fontsize=font_size_x_y, labelpad=10)
    ax2.set_ylabel(y_label_T, fontsize=font_size_x_y, labelpad=10)
    ax2.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
    ax2.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax2.minorticks_on()

    # set limits
    ax1.set_ylim(-5.5, 0.1)
    ax1.set_xlim(-5, 300)
    ax2.set_xlim(-5, 300)
    ax2.set_ylim(1.5, 7)

    epsilon = 1e-23

    # top plot
    ax1.plot(np.log10(sim_H[0] + epsilon), c="green", label=r'Ground truth')
    ax1.plot(np.log10(sim_H[1] + epsilon), c="green", label=r'_t=8')
    ax1.plot(np.log10(sim_H[2] + epsilon), c="green", label=r'_t=12')
    ax1.plot(np.log10(sim_H[3] + epsilon), c="green", label=r'_t=16')
    ax1.plot(np.log10(sim_H[4]) + epsilon, c="green", label=r'_t=20')

    ax1.plot(inf_H[0], c="orange", label=r'Inference')
    ax1.plot(inf_H[1], c="orange", label=r'_')
    ax1.plot(inf_H[2], c="orange", label=r'_')
    ax1.plot(inf_H[3], c="orange", label=r'_')
    ax1.plot(inf_H[4], c="orange", label=r'_')

    # bottom plot
    ax2.plot(np.log10(sim_T[0]), c="green", label=r'Ground truth')
    ax2.plot(np.log10(sim_T[1]), c="green", label=r'_t=8')
    ax2.plot(np.log10(sim_T[2]), c="green", label=r'_t=12')
    ax2.plot(np.log10(sim_T[3]), c="green", label=r'_t=16')
    ax2.plot(np.log10(sim_T[4]), c="green", label=r'_t=20')

    ax2.plot(inf_T[0], c="orange", label=r'Inference')
    ax2.plot(inf_T[1], c="orange", label=r'_t=4')
    ax2.plot(inf_T[2], c="orange", label=r'_t=4')
    ax2.plot(inf_T[3], c="orange", label=r'_t=4')
    ax2.plot(inf_T[4], c="orange", label=r'_t=4')

    # legends
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=font_size_legends)

    # annotation
    ax1.text(x=0.05, y=0.15,
             s=r'$\mathbf{t = (4,8,12,16,20)\; \mathrm{\textbf{Myr}}}$',
             transform=ax1.transAxes,
             fontsize=font_size_legends)

    # save fig
    path = 'test_run_4.pdf'
    fig.savefig(path)


if __name__ == "__main__":

    mlp1_compare_inference_examples()

