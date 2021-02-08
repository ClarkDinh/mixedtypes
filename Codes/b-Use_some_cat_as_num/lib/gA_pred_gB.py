
import numpy as np

try:
    from lib.normalize import get_pv_Xnorm_y
except Exception as e:
    from normalize import get_pv_Xnorm_y




def gA_pred_gB(df, pred_model, 
				inst_gA, inst_gB,
		    	tv, pv, rm_v=None):
	# df: dataframe. It includes both data instances of group A and group B.
	# pred_model: prediction model. The model with fix hyper parameter eg: alpha in Lasso, (alpha, gamma) in Kernel Ridge
	# inst_gA, inst_gB: instances to train (inst_gA) and instances to predict (inst_gB)

    # pv, X_train, y_train, inst_gA = get_pv_Xnorm_y(df=df, 
    # 	inst_idx=inst_gA, tv=tv, pv=pv, rm_v=rm_v)
    n_gA = len(inst_gA)

    inst_AB =  np.concatenate((inst_gA, inst_gB))
    pv, X, y, inst_AB = get_pv_Xnorm_y(df=df, 
    	inst_idx=inst_AB, tv=tv, pv=pv, rm_v=rm_v)


    X_train = X[:n_gA]
    y_train = y[:n_gA]
    X_test = X[n_gA:]
    y_test = y[n_gA:]


    pred_model.fit(X=X_train, y=y_train)

    y_test_pred = pred_model.predict(X_test)
    err = np.fabs(y_test - y_test_pred)

    return y_test, y_test_pred, err

    