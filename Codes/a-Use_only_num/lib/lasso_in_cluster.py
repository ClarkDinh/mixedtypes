import pandas as pd
from sklearn.model_selection import cross_val_predict

try:
    from lib.general_lib import *
    from lib.normalize import get_pv_Xnorm_y, get_Xnorm
    from lib.Isomap import Iso_map
    from lib.kr_parameter_search import get_estimator, alpha_search
    from lib.gA_pred_gB import gA_pred_gB
except Exception as e:
    from general_lib import *
    from normalize import get_pv_Xnorm_y, get_Xnorm
    from Isomap import Iso_map
    from kr_parameter_search import get_estimator, alpha_search
    from gA_pred_gB import gA_pred_gB

axis_font = {'fontname': 'serif', 'size': 16, 'labelpad': 8}
title_font = {'fontname': 'serif', 'size': 12}


def lasso_in_cluster(df, inst_idx, pv, tv, result_dir, n_cluster, rm_v=None, group_index=None,
                     lasso_revise=None,
                     params=None,
                     ):
    # # df, inst_idx: dataframe, instance indexes
    # # pv, tv, rm_v: predicting variables, target variable, remove variable
    # # n_cluster: predefined number of clusters

    pv, X, y, instance_name = get_pv_Xnorm_y(df=df, inst_idx=inst_idx,
                                             tv=tv, pv=pv, rm_v=rm_v)
    # fname_out = '{}/coefficient_result_n_cluster_{}.txt'.format(result_dir, n_cluster)
    # makedirs(file=fname_out)
    # file_out = open(fname_out, 'w')

    # if group predict == None then search for all compound, could be neglect
    if group_index is None:
        save_at = '{0}/isomap_{1}.pdf'.format(result_dir, n_cluster)
        group_index, x_after_isomap = Iso_map(X=X, params=params, annot_lbl=None)

        score_before, b, c, d = alpha_search(X=X, y=y,
                                             alpha_log_lb=params["alpha_log_lb"], alpha_log_ub=params["alpha_log_ub"],
                                             alpha_n_points=params["alpha_n_points"],
                                             predict_model=params["predict_model"], n_cv=params["n_cv"],
                                             n_times=params["n_times"])

        # print("score best before clustering:", score_before)
        # file_out.write("Score before clustering: {}".format(score_before))
        # file_out.write("=======================")
    else:
        x_after_isomap = False

    alpha_bests = []
    name_gs = []
    cluster_idx_array = []

    # # to find best alpha for each cluster
    for cluster_idx in range(n_cluster):
        name_g = instance_name[np.where(group_index == cluster_idx)]

        pv_g, X_g, y_g, name_g = get_pv_Xnorm_y(df=df, inst_idx=name_g,
                                                tv=tv, pv=pv, rm_v=rm_v)

        this_score, this_alpha_best, y_obs = alpha_search(X=X_g, y=y_g,
                                                                     alpha_log_lb=params["alpha_log_lb"],
                                                                     alpha_log_ub=params["alpha_log_ub"],
                                                                     alpha_n_points=params["alpha_n_points"],
                                                                     predict_model=params["predict_model"],
                                                                     n_cv=params["n_cv"], n_times=params["n_times"])

        alpha_bests.append(this_alpha_best)
        name_gs.append(name_g)
        tmp = [cluster_idx] * len(y_obs)
        cluster_idx_array = np.concatenate((cluster_idx_array, tmp), axis=0)

    y_obs_total = []
    y_pred_total = []

    score_gs = []
    mae_gs = []
    y_obs_gs = []
    y_pred_gs = []

    score_total_weight = 0

    # # to find score in each cluster and overall score
    # # could be included in previous section
    for cluster_idx in range(params["n_cluster"]):
        estimator = get_estimator(predict_model=params["predict_model"])
        estimator.alpha = alpha_bests[cluster_idx]
        name_in_this_cluster = name_gs[cluster_idx]

        pv_g, X_g, y_g, name_g = get_pv_Xnorm_y(df=df, inst_idx=name_in_this_cluster,
                                                tv=tv, pv=pv, rm_v=rm_v)
        this_group_idx = [cluster_idx] * len(y_g)
        y_pred_g = cross_val_predict(estimator=estimator, X=X_g, y=y_g, cv=params["n_cv"])

        this_score = score(y_obs=y_g, y_predict=y_pred_g, score_type='R2')
        this_err = error(y_obs=y_g, y_predict=y_pred_g, error_type='MAE')
        score_total_weight += this_score * len(name_in_this_cluster) / float(len(instance_name))

        y_obs_total = np.concatenate((y_obs_total, y_g), axis=0)
        y_pred_total = np.concatenate((y_pred_total, y_pred_g), axis=0)

        score_gs.append(this_score)
        mae_gs.append(this_err)
        y_obs_gs.append(y_g)
        y_pred_gs.append(y_pred_g)

    # print "name_in_this_cluster", name_in_this_cluster

    score_all = score(y_obs=y_obs_total, y_predict=y_pred_total)
    mae_all = error(y_obs=y_obs_total, y_predict=y_pred_total)

    if params["visualize"] and lasso_revise:
        # plot for each cluster
        for cluster_idx in range(params["n_cluster"]):

            this_score = score_gs[cluster_idx]
            this_err = mae_gs[cluster_idx]

            name_g = name_gs[cluster_idx]
            pv_g, X_g, y_g, name_g = get_pv_Xnorm_y(df=df, inst_idx=name_g,
                                                    tv=tv, pv=pv, rm_v=rm_v)
            estimator = get_estimator(predict_model=params["predict_model"])
            estimator.alpha = alpha_bests[cluster_idx]
            estimator.fit(X=X_g, y=y_g)
            c = estimator.coef_

            dtype = [('name', np.str_, 32), ('coeff', np.double)]
            var_coeff = []
            for i in range(len(c)):
                tmp = np.array((pv[i], c[i]), dtype=dtype)
                var_coeff.append(tmp)
            var_coeff = np.sort(var_coeff, order='coeff')  # sort from min to max

            # file_out.write("lasso_revise= {}".format(lasso_revise))
            # print("In group {0},  score best is: {1}".format(cluster_idx, this_score))
            # file_out.write("Cluster index: {}".format(cluster_idx))
            # file_out.write("Group score: {}".format(this_score))
            # file_out.write("Group error: {}".format(this_err))
            # file_out.write("Name in this cluster: {}".format(name_g))
            # file_out.write(str(var_coeff))
            # file_out.write("==================================================")

            # if this_score < 0.5:
            #     file_out.write("***addition for score = %f < 0.5".format(this_score))

    return group_index, alpha_bests, x_after_isomap, \
           score_all, mae_all, score_total_weight, \
           score_gs, mae_gs, y_obs_gs, y_pred_gs


def revise_KMEANs_by_LASSO(init_gidx=None, data_df=None,
                           params=dict({"df": None, "pv": None, "tv": None, "rm_v": None,
                                        "predict_model": 'Lasso', "out_dir": None,
                                        "alpha_log_lb": -4.0, "alpha_log_ub": 1.0, "alpha_n_points": 30,
                                        "n_cv": 10, "n_times": 3, "n_iterative": 30, "n_cluster": 3,
                                        "is_plot": False, "visualize": False})):
    pv = params["predicting_variable"]  # not have "s"
    tv = params["target_variable"]
    rm_v = params["remove_variable"]
    n_cluster = params["n_cluster"]
    if data_df is None:
        df = pd.read_csv(params["input_file"], index_col=0)
        # # normalize
        X_norm = get_Xnorm(X_matrix=df[pv].values)
        df[pv] = X_norm
    else:
        df = data_df
    n_iterative = params["n_iterative"]
    result_dir = params["out_dir"]
    inst_idx = df.index

    # new_df to keep trajectory, revise_df to print out the final decision
    new_df = pd.DataFrame(columns=(i for i in range(params["n_cluster"] + 1 + params["n_iterative"])), index=inst_idx)
    revise_df = pd.DataFrame(index=inst_idx, columns=np.arange(n_cluster + 1))

    for iterative in range(n_iterative):
        print("=============================")
        print("iterative number:", iterative)

        if iterative == 0:
            group_predict = None

        if iterative == 0 and init_gidx is not None:
            group_predict = init_gidx

        group_predict, alpha_bests, x_after_isomap, \
        score_total, this_best_error, score_total_weight, \
        score_gs, mae_gs, y_obs_gs, y_pred_gs = lasso_in_cluster(
            df=df, inst_idx=inst_idx,
            pv=pv, tv=tv, rm_v=rm_v, n_cluster=n_cluster,
            result_dir=result_dir, group_index=group_predict,
            lasso_revise=True,
            params=params)

        if iterative == 0 and init_gidx is None:
            xplot = x_after_isomap
        else:
            xplot = False

        # this_best_error means global error
        if iterative == 0 or this_best_error < best_err_out:  # score_total_weight < best_score_total_weight
            score_total_out = score_total
            score_std_out = np.std(score_gs)
            best_err_out = this_best_error
            score_total_weight_out = score_total_weight
            group_predict_out = group_predict

        # # for each cluster, establish prediction model
        # # predict to the rest of data points
        for cluster_idx in range(n_cluster):

            name_in_this_cluster = inst_idx[np.where(group_predict == cluster_idx)]
            name_beyond_this_cluster = inst_idx[np.where(group_predict != cluster_idx)]

            # # compute validation errors in each cluster
            _, X_in_cluster, y_in_cluster, _ = get_pv_Xnorm_y(df=df, inst_idx=name_in_this_cluster,
                                                              tv=tv, pv=pv, rm_v=rm_v)

            estimator = get_estimator(params["predict_model"])
            estimator.alpha = alpha_bests[cluster_idx]
            estimator.fit(X=X_in_cluster, y=y_in_cluster)

            y_predicted_in_cluster = cross_val_predict(estimator=estimator, X=X_in_cluster, y=y_in_cluster,
                                                       cv=params["n_cv"])
            errs = abs(y_predicted_in_cluster - y_in_cluster)

            for i_th, name in enumerate(name_in_this_cluster):
                new_df.loc[name, cluster_idx] = errs[i_th]
                if iterative == 0:
                    new_df.loc[name, n_cluster] = cluster_idx  # old value
                    revise_df.loc[name, n_cluster] = cluster_idx  # old value
                    revise_df.loc[name, cluster_idx] = errs[i_th]

            # # predict to other data points
            _, y_beyond_cluster_predict, err_beyond = gA_pred_gB(df=df, pred_model=estimator,
                                                                 inst_gA=name_in_this_cluster,
                                                                 inst_gB=name_beyond_this_cluster,
                                                                 tv=tv, pv=pv, rm_v=rm_v)

            for name_ith in range(len(name_beyond_this_cluster)):
                new_df.loc[str(name_beyond_this_cluster[name_ith]), cluster_idx] = err_beyond[name_ith]

        new_df[n_cluster + 1 + iterative] = new_df[range(n_cluster)].idxmin(axis='columns')

        new_group_predict = new_df[n_cluster + 1 + iterative]

        if (np.sum(np.abs(new_group_predict - group_predict)) == 0) or (iterative == (n_iterative - 1)):
            group_predict, alpha_bests, T, \
            score_total, best_error, score_total_weight, \
            score_gs, mae_gs, y_obs_gs, y_pred_gs = lasso_in_cluster(
                df=df, inst_idx=inst_idx,
                pv=pv, tv=tv, rm_v=rm_v, n_cluster=n_cluster,
                result_dir=result_dir, group_index=group_predict_out,
                lasso_revise=True,
                params=params)
            print("=============================")
            print("Converged. score: {0}, std: {1}, MAE: {2}, score weight: {3}".format(
                score_total, np.std(score_gs), best_error, score_total_weight))

            print("Best fit. score: {0}, std: {1}, MAE: {2}, score weight: {3}".format(
                score_total_out, score_std_out, best_err_out, score_total_weight_out))
            print("Take the best fit.")
            break
        group_predict = new_group_predict

    # if not assign group index manually, plot re clustering
    if params["visualize"] and xplot:
        fig = plt.figure(figsize=(8, 8))
        colors = ["red", "blue", "green", "yellow", "brown"]
        for i in range(len(xplot)):
            plt.annotate(inst_idx[i], xy=(xplot[i][0], xplot[i][1]), size=8)
            plt.scatter(xplot[i][0], xplot[i][1], s=130, alpha=0.8,
                        c=colors[group_predict_out[i]])

        plt.xlabel('Dimension 1', **axis_font)
        plt.ylabel('Dimension 2', **axis_font)
        plt.title(r'Re-clustering by LASSO', **title_font)
        plt.savefig('{0}/Re-clustering_{1}.pdf'.format(result_dir, n_cluster))
        release_mem(fig)

    # do not save any file out
    # new_df.to_csv("%s/traject_%d.csv" %(self.dir, self.n_cluster))
    # self.revise_df.to_csv("%s/revise_group_%d.csv" %(self.dir, self.n_cluster))

    return group_predict_out, score_total_out, score_std_out, best_err_out, score_total_weight_out
