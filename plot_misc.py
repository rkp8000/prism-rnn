import matplotlib.pyplot as plt
import numpy as np
from disp import colors, set_plot


def plot_12sum(t, us, rs, ys, y_hats, u_test, r_test, y_hat_test, y_lim=(-1.1, 1.1), sfx='', figsize=(12, 6)):
    fig, axs = plt.subplots(3, 3, figsize=figsize, tight_layout=True, sharex=True)
    
    # train
    for cu, (ax_col, u, r, y, y_hat) in enumerate(zip(axs.T, us, rs, ys, y_hats)):
        for cu_, u_ in enumerate(u.T):
            ax_col[0].axhline(cu_, color='gray', ls='--')
            ax_col[0].plot(t, cu_ + .9*u_, c='b')

        ax_col[1].plot(t, r[:, :20])

        for cy_, y_ in enumerate(y.T):
            ax_col[2].axhline(cy_, color='gray', ls='--')
            ax_col[2].plot(t, cy_ + .9*y_, c='k')
        for cy_, y_hat_ in enumerate(y_hat.T):
            ax_col[2].plot(t, cy_ + .9*y_hat_, c='r')

        set_plot(ax_col[0], y_lim=y_lim, y_label='u', title=f'Input {cu}{sfx}')
        set_plot(ax_col[1], y_label='r', title='Reservoir response')
        set_plot(ax_col[-1], y_lim=(-.1, 2.1), x_label='Time (s)', title='Output')

    # test
    ax_col = axs[:, 2]
    
    for cu_, u_ in enumerate(u_test.T):
        ax_col[0].axhline(cu_, color='gray', ls='--')
        ax_col[0].plot(t, cu_ + .9*u_, c='g')

    ax_col[1].plot(t, r_test[:, :20])

    for cy_, y_hat_ in enumerate(y_hat_test.T):
        ax_col[2].axhline(cy_, color='gray', ls='--')
        ax_col[2].plot(t, cy_ + .9*(y_hats[0][:, cy_] + y_hats[1][:, cy_]), c='k')
        ax_col[2].plot(t, cy_ + .9*y_hat_, c='r')

    set_plot(ax_col[0], y_lim=y_lim, y_label='u', title=f'Input {cu+2}{sfx}')
    set_plot(ax_col[1], y_label='r', title='Reservoir response')
    set_plot(ax_col[-1], x_label='Time (s)', title='Output')
    
    return fig, axs
    
    
def plot_12sum_no_targ(t, us, rs, y_hats, u_test, r_test, y_hat_test, y_lim=(-1.1, 1.1), sfx='', figsize=(12, 6)):
    fig, axs = plt.subplots(3, 3, figsize=figsize, tight_layout=True, sharex=True)
    
    # train
    for cu, (ax_col, u, r, y_hat) in enumerate(zip(axs.T, us, rs, y_hats)):
        for cu_, u_ in enumerate(u.T):
            ax_col[0].plot(t, cu_ + .9*u_, c='b')

        ax_col[1].plot(t, r[:, :20])

        for cy_, (y_hat_, color) in enumerate(zip(y_hat.T, colors)):
            ax_col[2].plot(t, y_hat_, c=color)

        set_plot(ax_col[0], y_label='u', title=f'Input {cu}{sfx}')
        set_plot(ax_col[1], y_label='r', title='Reservoir response')
        set_plot(ax_col[-1], x_label='Time (s)', title='Output')

    # test
    ax_col = axs[:, 2]
    
    for cu_, u_ in enumerate(u_test.T):
        ax_col[0].plot(t, cu_ + .9*u_, c='g')

    ax_col[1].plot(t, r_test[:, :20])

    for cy_, (y_hat_, color) in enumerate(zip(y_hat_test.T, colors)):
        # ax_col[2].plot(t, (y_hats[0][:, cy_] + y_hats[1][:, cy_])/np.sqrt(2), color=color, ls='--')
        ax_col[2].plot(t, y_hat_, color=color)

    set_plot(ax_col[0], y_label='u', title=f'Input {cu+2}{sfx}')
    set_plot(ax_col[1], y_label='r', title='Reservoir response')
    set_plot(ax_col[-1], x_label='Time (s)', title='Output')
    
    return fig, axs