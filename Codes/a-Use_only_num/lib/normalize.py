from sklearn import preprocessing


try:
    from lib.general_lib import *
except Exception as e:
    from general_lib import *



def get_Xnorm(X_matrix):

    min_max_scaler = preprocessing.MinMaxScaler()
    x_normed = min_max_scaler.fit_transform(X_matrix)
    #x_normed = X_matrix
    return x_normed

def get_pv_Xnorm_y(df, tv, pv=None, inst_idx=None, rm_v=None):
    # # inst_idx: instances index. "None" means taking all instances in data
    # # pv: predicting variable. "None" means taking all variable in data, remove tv later
    # # rm_v: remove variable. "None" means no remove any variables
    # # if inst_idx is not None, order of inst_idx is identical with order of df.index

    # # get pv
    if pv is None:
        pv = list(df.columns)
    if tv in pv:
        pv.remove(tv)

    # # remove rm_v from pv
    if rm_v is not None:
        if isinstance(rm_v, list):
            # if rm_v in list type: ["Z_T", "Z_R"]
            for rm_v_i in rm_v:
                if rm_v_i in pv:
                    pv.remove(rm_v_i)
        elif rm_v in pv: 
            # if rm_v in string type: "Z_T"
            pv.remove(rm_v_i)

    # # select X, y by inst_index
    if inst_idx is None:
        y = df[tv].values
        X = df[pv].values
        instance_name = df.index.values
    else:
        y = df.loc[inst_idx, tv].values
        X = df.loc[inst_idx, pv].values
        instance_name = inst_idx

    X = get_Xnorm(X_matrix=X)

    return pv, X, y, instance_name









