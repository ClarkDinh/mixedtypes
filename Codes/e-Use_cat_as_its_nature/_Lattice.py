#!/usr/bin/python
# -*- coding: utf-8 -*-
print(__doc__)
import matplotlib.offsetbox as offsetbox
import numpy as np

try:
    from lib.general_lib import *
    from lib.normalize import get_pv_Xnorm_y
    from lib.lasso_in_cluster import lasso_in_cluster
    from lib.plot import set_plot_configuration
    from lib.kr_parameter_search import get_estimator
    from lib.gA_pred_gB import gA_pred_gB
    from lib.kp_itbd import *
except Exception as e:
    from general_lib import *
    from normalize import get_pv_Xnorm_y
    from lasso_in_cluster import lasso_in_cluster
    from plot import set_plot_configuration
    from kr_parameter_search import get_estimator
    from gA_pred_gB import gA_pred_gB
    from kp_itbd import *


color_gs = {
    0: 'red',
    1: 'blue',
    2: 'orange',
    3: 'green',
    4: 'brown',
    5: 'violet',
    6: 'grey',
    7: 'purple'
}
size_point = 20
size_text = 10
alpha_point = 0.7
title_font = {'fontname': 'serif', 'size': 12}

def add_subplot(fig, row, col, nrows, ncols, x, y, color, tv=None, text=None):
    figure_position = row * ncols + col + 1

    ax = fig.add_subplot(nrows, ncols, figure_position)
    y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y,
                                                    tv=tv, size_fig='small')
    x_ref = np.linspace(y_min_plot, y_max_plot, 100)
    ax.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)
    ax.tick_params(axis='both', labelbottom=False, labelleft=False)

    ax.scatter(x, y, s=size_point, alpha=alpha_point, c=color)
    # for this_i, this_name in enumerate(name_in_this_cluster):
    #    plt.annotate(this_name, (y[this_i], y_predicted[this_i]), size=7)
    # add legend
    ob = offsetbox.AnchoredText(text, loc=4, prop=dict(fontsize=8))
    ax.add_artist(ob)
    return fig


def confusion_matrix(df, inst_idx, group_index, pv, tv, result_dir,
                     params=None):
    predict_model = params["predict_model"]
    rm_v = params["remove_variable"]
    visualize = True
    params["visualize"] = visualize
    n_cv = params["n_cv"]
    n_times = params["n_cv"]
    # get pv, Xnorm and y for all
    pv, X, y, instance_name = get_pv_Xnorm_y(df=df, inst_idx=inst_idx, tv=tv, pv=pv, rm_v=rm_v)

    # get n_cluster
    n_cluster = max(group_index) + 1
    params["n_cluster"] = n_cluster

    predict_df = pd.DataFrame(index=instance_name,
                              columns=['g_{0}'.format(k) for k in range(n_cluster)])

    group_index, alpha_bests, x_after_isomap, \
    score_all_grps, err_all_grps, score_total_weight, \
    score_gs, error_gs, y_obs_gs, y_pred_gs = lasso_in_cluster(df=df, inst_idx=inst_idx,
                                                               pv=pv, tv=tv, result_dir=result_dir, n_cluster=n_cluster,
                                                               group_index=group_index, lasso_revise=None,
                                                               params=params)

    score_matrix = np.empty([n_cluster, n_cluster])
    if visualize:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        title = "Overview: estimator model: {0} \n R2: {1}, MAE: {2}, R2 weight: {3}".format(
            predict_model, round(score_all_grps, 3),
            round(err_all_grps, 3), round(score_total_weight, 3))

        plt.title(title, **title_font)
        plt.box(on=None) #Tai insert

    for g in range(n_cluster):
        name_g = instance_name[np.where(group_index == g)]
        pv_g, X_g, y_g, name_g = get_pv_Xnorm_y(df=df, inst_idx=name_g,
                                                tv=tv, pv=pv, rm_v=rm_v)

        estimator = get_estimator(predict_model=predict_model)
        estimator.alpha = alpha_bests[g]
        estimator.fit(X=X_g, y=y_g)
        y_pred_g = estimator.predict(X=X_g)
        # print(len(name_g), len(df.index))
        predict_df.loc[name_g, 'g_{0}'.format(g)] = y_pred_g

        score_g2g = score(y_obs=y_g, y_predict=y_pred_g, score_type='R2')
        error_g2g = error(y_obs=y_g, y_predict=y_pred_g)  # score

        if visualize:
            text = "R2: {0} \nMAE: {1}".format(round(score_g2g, 3),
                                                    round(error_g2g, 3))
            fig = add_subplot(fig=fig, row=g, col=g, nrows=n_cluster, ncols=n_cluster,
                              x=y_g, y=y_pred_g, color=color_gs[g], tv=tv, text=text)

        score_matrix[g][g] = score_g2g
        # to prepare loop for others cluster diff 2 g
        gen = (x for x in range(n_cluster) if x != g)

        for k in gen:
            name_k = instance_name[np.where(group_index == k)]

            # pred_model_g = estimator
            pred_model_g = get_estimator(predict_model=predict_model)
            pred_model_g.alpha = alpha_bests[g]

            y_k, y_pred_k, err_out = gA_pred_gB(df=df, pred_model=pred_model_g,
                                                inst_gA=name_g, inst_gB=name_k,
                                                tv=tv, pv=pv, rm_v=rm_v)

            predict_df.at[name_k, 'g_{0}'.format(k)] = y_pred_k
            score_g2k = score(y_obs=y_k, y_predict=y_pred_k, score_type='R2')
            error_g2k = error(y_obs=y_k, y_predict=y_pred_k)

            score_matrix[g][k] = score_g2k

            if visualize:
                text = "R2: {0} \nMAE: {1}".format(round(score_g2k, 3),
                                                        round(error_g2k, 3))

                fig = add_subplot(fig=fig, row=g, col=k,
                                  nrows=n_cluster, ncols=n_cluster,
                                  color=color_gs[k],
                                  x=y_k, y=y_pred_k,
                                  tv=tv, text=text)
    if visualize:
        save_at = "{0}/sep_nc={1}.pdf".format(result_dir, n_cluster)
        makedirs(save_at)
        plt.savefig(save_at)
        print("Save at", save_at)
        release_mem(fig)

    return score_matrix, group_index, score_all_grps
    # return score_matrix, group_index

def _cal_global_attr_freq(X):
    # global_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of attributes.
    npoints, nattrs = X.shape
    global_attr_freq = [defaultdict(float) for _ in range(nattrs)]

    for ipoint, curpoint in enumerate(X):
        for iattr, curattr in enumerate(curpoint):
            global_attr_freq[iattr][curattr] += 1.
    for iattr in range(nattrs):
        for key in global_attr_freq[iattr].keys():
            global_attr_freq[iattr][key] /= npoints

    return global_attr_freq

def run(argv):
    global global_attr_freq
    file_name = "Lattice.csv"
    tv = "Lattice_Constant"
    cat_cols=["Atomic_Number_A", "Atomic_Number_B", "atom_A", "atom_B",
     "group_A", "group_B", "period_A", "period_B", "group_index"]
    num_clusters = 3
    input_dir = "input"
    result_dir = "output"
    init = "Cao"
    n_init = 100
    verbose = 1
    input_dir = "input"
    result_dir = "output"

    source_file = "{0}/{1}".format(input_dir,file_name)
    only_name = file_name.replace(".csv","")
    save_result_dir = result_dir + "/{0}/sep_nc={1}_init={2}".format(only_name, num_clusters,init)
    makedirs(save_result_dir)
    gidx_file = "{0}/sep_nc={1}.csv".format(save_result_dir,num_clusters)
    makedirs(gidx_file)

    data_df = pd.read_csv(source_file, index_col=0)
    data_clustering_df = data_df.loc[:, cat_cols]
    X = np.array(data_clustering_df)
    X, enc_map = encode_features(X)
    global_attr_freq = _cal_global_attr_freq(X)
    clusterers = KRepresentative(n_clusters=num_clusters, init=init, n_init = n_init, global_attr_freq=global_attr_freq,
                                 verbose=verbose)
    clusterers.fit_predict(X)
    cluster_labels =  clusterers.labels_
    # print(cluster_labels)
    log_df = pd.DataFrame()
    log_df.loc[:,"compound"] = data_df.index
    log_df.loc[:,"result0"] = cluster_labels
    log_df.to_csv(gidx_file,index=False)

    gidx_df = pd.read_csv(gidx_file, index_col=0)
    group_index = gidx_df["result0"].values

    data_df = data_df.drop(cat_cols, axis = 1)
    pv = list(data_df.columns)
    pv.remove(tv)
    # print(pv)

    confusion_matrix(df=data_df, inst_idx=data_df.index, group_index=group_index,
                                       pv=pv, tv=tv, result_dir=save_result_dir,
                                       params=dict({"predict_model": 'Lasso',
                                                    "remove_variable": None, "n_cv": 10, "n_times": 50,
                                                    "alpha_log_ub": -1, "alpha_log_lb": -5, "alpha_n_points": 30}))
    print("Finish!!!!!")

                                             
if __name__ == "__main__":
    run(sys.argv[1:])