import numpy as np
import matplotlib.pyplot as plt
import time, gc, os
import pandas as pd
import seaborn as sns
import matplotlib

axis_font = {'fontname': 'serif', 'size': 16, 'labelpad': 8}
title_font = {'fontname': 'serif', 'size': 12}
size_point = 12
size_text = 5
alpha_point = 0.3
n_neighbor = 3


def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()


def ax_setting():
    plt.style.use('default')
    plt.tick_params(axis='x', which='major', labelsize=13)
    plt.tick_params(axis='y', which='major', labelsize=13)
    plt.tight_layout(pad=1.1)


def makedirs(file):
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))


def joint_plot(x, y, xlabel, ylabel, save_at):
    fig = plt.figure(figsize=(20, 20))
    # sns.set_style('ticks')
    sns.plotting_context(font_scale=1.5)
    this_df = pd.DataFrame()

    this_df[xlabel] = x
    this_df[ylabel] = y

    ax = sns.jointplot(this_df[xlabel], this_df[ylabel],
                       kind="kde", shade=True,
                       ).set_axis_labels(xlabel, ylabel)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.xlabel(r'%s' %xlabel, **axis_font)
    # plt.ylabel(r'%s' %ylabel, **axis_font)
    # plt.title(title, **self.axis_font)

    # plt.set_tlabel('sigma', **axis_font)
    # ax_setting()
    save_file = "{0}_joint.pdf".format(save_at)
    plt.tight_layout()
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    plt.savefig(save_file)

    print("Save file at:", "{0}".format(save_file))
    release_mem(fig)


def scatter_plot(x, y, save_file=None, x_label='x', y_label='y', annot_lbl=None, lbl=None,
                 mode='scatter', sigma=None, title=None,
                 interpolate=False, color='blue', ax=None, linestyle='-.', marker='o'):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))

    if 'scatter' in mode:
        plt.scatter(x, y, s=50, alpha=0.8, c=color, label=lbl, cmap='jet',
                    vmin=min(color), vmax=max(color))  # brown
        # plt.colorbar()

        if isinstance(color, str):
            plt.scatter(x, y, s=2, alpha=0.1, c=color, label=lbl)  # brown

    if 'line' in mode:
        plt.plot(x, y, marker=marker, linestyle=linestyle, color=color,
                 alpha=1.0, label=lbl, markersize=5, mfc='none')

    if 'error' in mode and sigma is not None:
        plt.fill_between(x, y - sigma, y + sigma, color=color, alpha=0.3)
    # plt.errorbar(x, y, sigma, color=color, alpha=0.5)

    if annot_lbl is not None:
        for i in range(len(x)):
            if (i % 50 == 0):
                plt.annotate(annot_lbl[i], xy=(x[i], y[i]), size=size_text)
    if interpolate:
        # work only for 2D
        x_new = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x)) / 1000.0)
        # from scipy import interpolate
        # argsort_x = np.argsort(x)
        # print (np.sort(x), y[argsort_x])
        # cs = interpolate.CubicSpline(x[argsort_x], y[argsort_x], bc_type='natural')
        # plt.plot(x_new, cs(x_new), color='orange',label='interpolate', alpha=0.6 )
        # plt.plot(x_new, cs(x_new, 1), color='orange',label='1st derivative' )
        # plt.plot(x_new, cs(x_new, 2), color='orange',label='2nd derivative' )
        # plt.plot(x_new, cs(x_new, 3), color='orange',label='3rd derivative' )

        from sklearn import gaussian_process
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
        kernel = ConstantKernel() + RBF(length_scale=0.001) + WhiteKernel(noise_level=0.5)  # length_scale=0.01, nu=3/2
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        # gp = gaussian_process.GaussianProcessRegressor()

        gp.fit(x.reshape(-1, 1), y)
        y_pred, sigma = gp.predict(x_new.reshape(-1, 1), return_std=True)

        plt.plot(x_new.reshape(-1, 1), y_pred, '-.', color='red', alpha=1.0, label="Interpolation")
        plt.errorbar(x_new.reshape(-1, 1), y_pred, sigma, color='orange', alpha=0.1, label="sigma")
    # plt.title(str(kernel), **title_font)
    # print (m.predict_f_samples(xx, 1).squeeze())
    # realizations = np.vstack(realizations)

    plt.ylabel(y_label, **axis_font)
    plt.xlabel(x_label, **axis_font)
    if title is not None:
        plt.title(title, **title_font)
    ax_setting()

    if ax is None:
        plt.legend(prop={'size': 16})
        makedirs(save_file)
        plt.savefig(save_file)
        release_mem(fig=fig)


def set_plot_configuration(x, y, tv, size_fig=None, ax=None):
    y_min = min([min(x), min(y)])
    y_max = max([max(x), max(y)])
    y_mean = (y_max + y_min) / 2.0
    y_std = (y_max - y_mean) / 2.0
    y_min_plot = y_mean - 2.4 * y_std
    y_max_plot = y_mean + 2.4 * y_std

    # threshold = 0.1
    # plt.plot(x_ref, x_ref * (1 + threshold), 'g--', label=r'$\pm 10 \%$')
    # plt.plot(x_ref, x_ref * (1 - threshold), 'g--', label='')
    if ax is None:
        plt.ylim([y_min_plot, y_max_plot])
        plt.xlim([y_min_plot, y_max_plot])
    else:
        ax.set_ylim([y_min_plot, y_max_plot])
        ax.set_xlim([y_min_plot, y_max_plot])
    if size_fig == None:
        plt.ylabel('{0} predicted'.format(tv), **axis_font)
        plt.xlabel('{0} observed'.format(tv), **axis_font)
        plt.legend(loc=2, fontsize='small')

    return y_min_plot, y_max_plot


def plot_regression(x, y, tv, dimensional, n_cluster=None, group_index=None, name=None, ):
    fig = plt.figure(figsize=(8, 8))
    if group_index is None:
        plt.scatter(x, y, s=size_point, alpha=alpha_point, c='blue', label=None)
        y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y,
                                                        tv=tv, dimensional=dimensional, size_fig=" ")
        x_ref = np.linspace(y_min_plot, y_max_plot, 100)
        plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)

        # if name is not None:
        #    for i in range(len(name)):
        #        plt.annotate(str(name[i]), xy=(x[i], y[i]), size=size_text)


    else:
        if isinstance(group_index, int) == True:
            plt.scatter(x, y, c=option[group_index], s=size_point,
                        alpha=alpha_point, label='Group %d' % (group_index + 1))
            if name is not None:
                for i in range(len(x)):
                    # only for lattice_constant problem, 1_Ag-H, 10_Ag-He
					# if tmp_check_name(name=name[i]):
                    #    reduce_name = str(name[i]).split('_')[1]
                    #    plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)

                    plt.annotate(name[i], xy=(x[i], y[i]), size=size_text)

            y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y,
                                                            tv=tv, dimensional=dimensional, )
            x_ref = np.linspace(y_min_plot, y_max_plot, 100)
            plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)
        else:
            for this_group in range(config["n_cluster"]):
                # print this_group, np.where(group_index == this_group)
                x_plot = x[this_group]
                y_plot = y[this_group]

                plt.scatter(x_plot, y_plot, s=size_point, alpha=alpha_point,
                            c=option[this_group], label='Group %d' % (this_group + 1))
                if name is not None:
                    for i in range(len(x_plot)):
                        plt.annotate(name[this_group][i], xy=(x_plot[i], y_plot[i]), size=size_text)

            x_all = [item for sublist in x for item in sublist]
            y_all = [item for sublist in y for item in sublist]
            y_min_plot, y_max_plot = set_plot_configuration(x=x_all, y=y_all, config=config)
            x_ref = np.linspace(y_min_plot, y_max_plot, 100)

            plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)
    return fig


def get_color(source_df, col_clr):
    col_array = source_df[col_clr].values

    cnorm = matplotlib.cm.ScalarMappable()
    cnorm.set_clim(vmin=None, vmax=None)

    to_rgba = cnorm.to_rgba(col_array)
    # cmap = matplotlib.cm.get_cmap('Spectral',
    # 		vmin=min(col_array), vmax=max(col_array))

    col_names = dict({k: v for k, v in zip(source_df.index, to_rgba)})
    # for inst, col_val in zip(source_df.index, col_array):
    # 	col_names[inst] = cnorm.inverse(col_val)

    return col_names




