
import sys, os, yaml
import pandas as pd


def load_config(cfg_file=None):
    # print(len(sys.argv))
    if cfg_file is None:
        # works on local, config file (cfg_file) shows as 1st arg
        if len(sys.argv) != 2:
            print('Error !!!')
            print('Please give an argument which is configuration file\'s path')
            print('Example:')
            print('\tpython regression-based.py regression-based.yaml')
            quit()
        config_file = sys.argv[1]
        print(config_file)

    else:
        # works on iselohn, config file (cfg_file) read as path in arg
        config_file = cfg_file

    with open(config_file, 'r') as stream:
        try:
            content = yaml.load(stream)
            print (content)
            return content
        except yaml.YAMLError as exc:
            print(exc)
            exit()



def remove_tv_rv(input_var_dict, var_list, all_variables):
    tv = input_var_dict["target_variable"]
    if "remove_variable" in input_var_dict and input_var_dict["remove_variable"] is not None:
        for v in input_var_dict["remove_variable"]:
            var_list.remove(v)

    if tv in var_list:
        var_list.remove(tv)

    for v in var_list:
        if not v in all_variables:
            print ("Error!!!")
            print ("Variable \'{0}\' in {1} not found in input file.".format(v, var_list))
            quit()

    return var_list



def get_predict_variables(input_dir, input_file, input_var_dict):
    # read predicting_variables
    input_df = pd.read_csv(input_file, index_col=0)
    all_variables = input_df.columns
    predicting_variables = []
    if "predicting_variables" in input_var_dict and input_var_dict["predicting_variables"] is not None:
        pv_file = input_dir + '/' + str(input_var_dict["predicting_variables"])

        if os.path.isfile(pv_file):
            # multiple combination of predicting variables
            df_pv = pd.read_csv(pv_file, index_col=0)
            try:
                labels = list(df_pv["label"])
                predicting_variables = []
                for item in labels:
                    var_list = list(str(item).split("|"))
                    pv = remove_tv_rv(input_var_dict=input_var_dict,
                                      var_list=var_list,
                                      all_variables=all_variables)
                    predicting_variables.append(pv)
            except:
                print ("Error! No \"label\" column in predicting variables file")
                quit()
        else:
            # only 1 combination
            var_list = input_var_dict["predicting_variables"]
            pv = remove_tv_rv(input_var_dict=input_var_dict,
                              var_list=var_list,
                              all_variables=all_variables)
            predicting_variables.append(pv)
    else:
        var_list = list(all_variables)
        pv = remove_tv_rv(input_var_dict=input_var_dict,
                          var_list=var_list,
                          all_variables=all_variables)
        predicting_variables.append(pv)

    return predicting_variables


def check_consistent(config_dict):

    if config_dict["initial_method"] is 'manual':
        if config_dict["n_cluster"] != len(config_dict["assign"]):
            print ("Error!!!")
            print ("Manual group assignment:", config_dict["assign"])
            print ("n_cluster:", config_dict["n_cluster"])
            print ("Number of value in manual group assignment differ from Option n_cluster.")
            quit()


def read_config(config):
    config_name = config["config_name"]

    # general, input, output
    general = config["general"]
    directory = general["directory"]
    input_dir = str(directory["input_dir"])
    input_plain = str(directory["input_file"])
    if input_dir is None:
        input_dir = os.getcwd()
    file_name = str(directory["input_file"])
    input_file = input_dir + '/' + str(directory["input_file"])
    if not os.path.isfile(input_file):
        print ("Error! Input file not found.")
        quit()

    out_dir = str(directory["out_dir"])
    out_extend = directory["out_extend"]
    out_conclusion = str(directory["out_conclusion"])
    test_file = input_dir + '/' + str(directory["test_file"])
    pred_out_file = directory["pred_out_file"]

    input_var_dict = general["variables"]
    target_variable = input_var_dict["target_variable"]
    dimensional = input_var_dict["dimensional"]
    remove_variable = input_var_dict["remove_variable"]

    predicting_variables = get_predict_variables(input_dir=input_dir,
                                                 input_file=input_file,
                                                 input_var_dict=input_var_dict)

    n_cluster = general["n_cluster"]
    top_k = general["top_k"]
    n_clusters = general["n_clusters"]
    n_iterative = general["n_iterative"]

    predict_model = general["predict_model"]

    alpha_search = general["alpha_search"]
    alpha_n_points = alpha_search["alpha_n_points"]
    alpha_log_lb = alpha_search["alpha_log_lb"]
    alpha_log_ub = alpha_search["alpha_log_ub"]

    evaluation = general["evaluation"]
    n_cv = evaluation["n_cv"]
    n_times = evaluation["n_times"]

    score = evaluation["score"]
    error = evaluation["error"]
    cv_krr = evaluation["cv_krr"]

    initial_state = general["initial_state"]
    initial_method = initial_state["method"]
    # auto assign
    n_neighbors = initial_state["auto"]["n_neighbors"]
    n_components = initial_state["auto"]["n_components"]

    # manual assign
    criteria_column = initial_state["manual"]["column"]
    assign = initial_state["manual"]["assign"]

    if initial_method == 'manual':
        n_neighbor = None
        n_components = None
    else:
        criteria_column = None
        assign = None

    svm_variable = general["svm_variable"]
    visualize = general["visualize"]

    RBC_best_time = general["RBC_best_time"]
    RBC_size_column = general["RBC_size_column"]
    RBC_ntimes = general["RBC_ntimes"]

    # source file for only train-test split
    if "source_file" in config:
        source_file = config["source_file"]
    else:
        source_file = input_file

    function = config["function"]
    RBC_score_var = function["RBC_score_var"]
    manual_assign_group_index = function["manual_assign_group_index"]
    analyze_RBC_score_var = function["analyze_RBC_score_var"]
    predict_test_rrbc = function["predict_test_rrbc"]


    # plot_score_depend_ncluster = function["plot_score_depend_ncluster"]
    # combine_clustering_work = function["combine_clustering_work"]
    # plot_post_analysis = function["plot_post_analysis"]
    # confusion_matrix_2 = function["confusion_matrix_2"]
    # coefficient_distance = function["coefficient_distance"]
    # get_comparision_linear_nonlinear = function["get_comparision_linear_nonlinear"]
    # svm_param_search = function["svm_param_search"]
    # predict_test_krr = function["predict_test_krr"]
    # decision_tree = function["decision_tree"]


    config_dict = {
        # Directories
        "config_name": config_name,
        "input_file": input_file,
        "file_name": file_name,
        "input_plain": input_plain,
        "input_dir": input_dir,
        "out_dir": out_dir,
        "out_extend": out_extend,
        "test_file": test_file,
        "pred_out_file": pred_out_file,
        "out_conclusion": out_conclusion,

        # Variables
        "target_variable": target_variable,
        "dimensional": dimensional,
        "predicting_variables": predicting_variables,
        "remove_variable": remove_variable,
        "svm_variable": svm_variable,

        # General setting
        "n_cluster": n_cluster,
        "top_k": top_k,
        "n_clusters": n_clusters,
        "n_iterative": n_iterative,
        "predict_model": predict_model,
        "alpha_n_points": alpha_n_points,
        "alpha_log_lb": alpha_log_lb,
        "alpha_log_ub": alpha_log_ub,
        "n_cv": n_cv,
        "n_times": n_times,
        "score": score,
        "error": error,
        "cv_krr": cv_krr,

        # Initial state method
        "initial_method": initial_method,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "criteria_column": criteria_column,
        "assign": assign,
        "visualize": visualize,

        # Best time of random sampling
        "RBC_best_time": RBC_best_time,
        "RBC_size_column": RBC_size_column,
        "RBC_ntimes": RBC_ntimes,

        # source-file for only train-test split
        "source_file": source_file
    }

    function_dict = {
            "RBC_score_var": RBC_score_var,
        "manual_assign_group_index": manual_assign_group_index,
        "analyze_RBC_score_var": analyze_RBC_score_var,
        "predict_test_rrbc": predict_test_rrbc

        # "plot_score_depend_ncluster": plot_score_depend_ncluster,
        # "combine_clustering_work": combine_clustering_work,
        # "plot_post_analysis": plot_post_analysis,
        # "confusion_matrix_2": confusion_matrix_2,
        # "coefficient_distance": coefficient_distance,
        # "get_comparision_linear_nonlinear": get_comparision_linear_nonlinear,
        # "svm_param_search": svm_param_search,
        # "predict_test_krr": predict_test_krr,
        # "decision_tree": decision_tree,
    }

    check_consistent(config_dict=config_dict)


    return config_dict, function_dict

































