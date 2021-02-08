import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, scale
from sklearn.linear_model import Lasso, Ridge, Lars, LinearRegression

try:
    from lib.normalize import get_pv_Xnorm_y, get_Xnorm
except Exception as e:
    from normalize import get_pv_Xnorm_y, get_Xnorm

#############################################################
#   Generate n_times trials on learning model from data
#   Data is splited into trainning set and test set
#   Training set is used for trainning model
#   Test set is used for prediction
#   Return
def CV_predict(model, X, y, n_folds=3, n_times=3):

    if (n_folds <= 0) or (n_folds > len(y)):
        n_folds = len(y)
        n_times = 1

    y_predicts = []
    for i in range(n_times):
        indexes = np.random.permutation(range(len(y)))

        kf = KFold(n_splits=n_folds)

        y_cv_predict = []
        cv_test_indexes = []
        cv_train_indexes = []
        for train, test in kf.split(indexes):
            # cv_train_indexes += list(indexes[train])
            cv_test_indexes += list(indexes[test])

            X_train, X_test = X[indexes[train]], X[indexes[test]]
            y_train, Y_test = y[indexes[train]], y[indexes[test]]

            model.fit(X_train, y_train)

            # y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_test)
            y_cv_predict += list(y_test_predict)

        cv_test_indexes = np.array(cv_test_indexes)
        rev_indexes = np.argsort(cv_test_indexes)

        y_cv_predict = np.array(y_cv_predict)

        y_predicts += [y_cv_predict[rev_indexes]]

    y_predicts = np.array(y_predicts)

    return y_predicts



def CV_predict_score(model, X, y, n_folds=3, n_times=3):

    if (n_folds <= 0) or (n_folds > len(y)):
        n_folds = len(y)
        n_times = 1

    y_predicts = []
    scores = []
    errors = []
    for i in range(n_times):
        y_predict = CV_predict(model=model, X=X, y=y, n_folds=n_folds, n_times=1)
        # n_times = 1 then the result has only 1 y_pred array
        y_predict = y_predict[0]
        this_r2 = r2_score(y_true=y, y_pred=y_predict)
        this_err = mean_absolute_error(y_true=y, y_pred=y_predict)
        scores.append(this_r2)
        errors.append(this_err)

    return np.mean(scores), np.std(scores), np.mean(errors), np.std(errors)


def kernel_ridge_parameter_search(X, y_obs, kernel='rbf',
                                  n_folds=3, n_times=3):
    # parameter initialize
    gamma_log_lb = -2.0 # old -2.0
    gamma_log_ub = 2.0 # old 2.0
    alpha_log_lb = -4.0 # old -4.0
    alpha_log_ub = -1.0 # old 1.0
    n_steps = 20
    n_rounds = 4
    alpha = 1
    gamma = 1
    lb = 0.8
    ub = 1.2
    n_instance = len(y_obs)

    if (n_folds <= 0) or (n_folds > n_instance):
        n_folds = n_instance
        n_times = 1

    # Start
    for i in range(n_rounds):
        scores_mean = []
        scores_std = []
        gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
        for gamma in gammas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            cv_scores = list(map(lambda y_predict: r2_score(
                y_obs, y_predict), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        gamma = gammas[best_index]
        gamma_log_lb = np.log10(gamma * lb)
        gamma_log_ub = np.log10(gamma * ub)
        scores_mean = []
        scores_std = []
        alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
        for alpha in alphas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            cv_scores = list(map(lambda y_predict: r2_score(
                y_obs, y_predict), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        alpha = alphas[best_index]
        alpha_log_lb = np.log10(alpha * lb)
        alpha_log_ub = np.log10(alpha * ub)

    return alpha, gamma, scores_mean[best_index], scores_std[best_index]


def kernel_ridge_parameter_search_boost(X, y_obs, kernel='rbf', n_folds=3,
                                        n_times=3, n_dsp=160, n_spt=5):
    """
    """
    # parameter initialize
    gamma_log_lb = -2.0
    gamma_log_ub = 2.0
    alpha_log_lb = -4.0
    alpha_log_ub = 1.0
    n_steps = 10
    n_rounds = 4
    alpha = 1
    gamma = 1
    lb = 0.8
    ub = 1.2
    n_instance = len(y_obs)
    if (n_dsp > n_instance) or (n_dsp <= 0):
        n_dsp = n_instance
        n_spt = 1

    if (n_folds <= 0) or (n_folds > n_instance):
        n_folds = n_instance
        n_times = 1

    for i in range(n_rounds):
        # Searching for Gamma
        gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
        best_gammas = []
        for _ in range(n_spt):
            scores_mean = []
            scores_std = []

            indexes = np.random.permutation(range(n_instance))
            X_sample = X[indexes[:n_dsp]]
            y_obs_sample = y_obs[indexes[:n_dsp]]

            for gamma in gammas:
                k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
                y_sample_predict = CV_predict(k_ridge, X_sample, y_obs_sample,
                                              n_folds=n_folds, n_times=n_times)
                cv_scores = map(lambda y_sample_predict: r2_score(
                    y_obs_sample, y_sample_predict), y_sample_predict)

                scores_mean += [np.mean(cv_scores)]
                scores_std += [np.std(cv_scores)]

            best_index = np.argmax(scores_mean)
            gamma = gammas[best_index]

            best_gammas += [gamma]

        best_gammas = np.array(best_gammas)
        gamma = np.mean(best_gammas)

        gamma_log_lb = np.log10(gamma * lb)
        gamma_log_ub = np.log10(gamma * ub)

        # Searching for Alpha
        alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
        best_alphas = []
        for _ in range(n_spt):
            scores_mean = []
            scores_std = []

            indexes = np.random.permutation(range(n_instance))
            X_sample = X[indexes[:n_dsp]]
            y_obs_sample = y_obs[indexes[:n_dsp]]

            for alpha in alphas:
                k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
                y_sample_predict = CV_predict(k_ridge, X_sample, y_obs_sample,
                                              n_folds=n_folds, n_times=n_times)
                cv_scores = map(lambda y_sample_predict: r2_score(
                    y_obs_sample, y_sample_predict), y_sample_predict)

                scores_mean += [np.mean(cv_scores)]
                scores_std += [np.std(cv_scores)]

            best_index = np.argmax(scores_mean)
            alpha = alphas[best_index]

            best_alphas += [alpha]

        best_alphas = np.array(best_alphas)
        alpha = np.mean(best_alphas)

        alpha_log_lb = np.log10(alpha * lb)
        alpha_log_ub = np.log10(alpha * ub)

    return alpha, gamma, scores_mean[best_index], scores_std[best_index]


def kernel_ridge_cv(data, target_variable, predicting_variables,
                    kernel='rbf', n_folds=10, n_times=100):
    """ Alias "kr"
    """

    if target_variable in predicting_variables:
        predicting_variables.remove(target_variable)

    X = data.as_matrix(columns=predicting_variables)
    y_obs = data.as_matrix(columns=(target_variable,)).ravel()

    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(X)
    min_max_scaler_y = MinMaxScaler()
    y_obs = min_max_scaler_y.fit_transform(y_obs)

    best_alpha, best_gamma, best_score, best_score_std = \
        kernel_ridge_parameter_search(X, y_obs, kernel=kernel,
                                      n_folds=n_folds, n_times=n_times)

    return best_alpha, best_gamma, best_score, best_score_std


def kernel_ridge_cv_boost(data, target_variable, predicting_variables,
                          kernel='rbf', n_folds=10, n_times=100,
                          n_dsp=160, n_spt=5):
    """ Alias "kr_boost"
    """

    if target_variable in predicting_variables:
        predicting_variables.remove(target_variable)

    X = data.as_matrix(columns=predicting_variables)
    y_obs = data.as_matrix(columns=(target_variable,)).ravel()

    min_max_scaler_X = MinMaxScaler()
    X = min_max_scaler_X.fit_transform(X)
    min_max_scaler_y = MinMaxScaler()
    y_obs = min_max_scaler_y.fit_transform(y_obs)

    best_alpha, best_gamma, best_score, best_score_std = \
        kernel_ridge_parameter_search_boost(X, y_obs, kernel=kernel,
                                            n_folds=n_folds, n_times=n_times,
                                            n_dsp=n_dsp, n_spt=n_spt)

    return best_alpha, best_gamma, best_score, best_score_std



def get_estimator(predict_model='Lasso'):
    
    if predict_model == 'Lasso':
        estimator = Lasso()
    elif predict_model == 'Ridge':
        estimator = Ridge()
    elif predict_model == 'Lars':
        estimator = Lars()

    return estimator

def alpha_search(X, y, alpha_log_lb=-4.0, alpha_log_ub=1.0, alpha_n_points=30, predict_model='Lasso',
    n_cv=10, n_times=3):


    alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=alpha_n_points)


    score_depend_alpha = []
    err_depend_alpha = []
    coeffs_depend_alphas = []

    estimator = get_estimator(predict_model=predict_model)
    for alpha in alphas:
        estimator.alpha = alpha

        score, score_std, err, err_std = CV_predict_score(model=estimator, X=X, y=y, 
            n_folds=n_cv, n_times=n_times)
        score_depend_alpha.append(score)
        err_depend_alpha.append(err)
        
        estimator.fit(X=X, y=y)
        coeffs_depend_alphas.append(estimator.coef_)

    idx_max = np.argmax(score_depend_alpha)
    idx_err_min = np.argmin(err_depend_alpha)

    alpha_best_score = alphas[idx_max]
    alpha_best_err = alphas[idx_err_min]

    score_best = score_depend_alpha[idx_max]
    err_best = err_depend_alpha[idx_err_min]



    estimator.alpha = alpha_best_score
    # y_predicted = cross_val_predict(estimator=estimator, X=X, y=y, cv=n_cv)


    # return score_best, alpha_best_score, y, y_predicted
    return score_best, alpha_best_score, y
