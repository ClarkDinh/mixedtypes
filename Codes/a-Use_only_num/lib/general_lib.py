import yaml
import sys
import gc
import os
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error


def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()

def get_basename(filename):
    head, tail = ntpath.split(filename)

    basename = os.path.splitext(tail)[0]
    return tail

def makedirs(file):
	print(os.path.dirname(file))
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))
	# if not os.path.exists(dirName):
	# 	os.makedirs(dirName)
	# 	print("Directory " , dirName ,  " Created ")
	# else:
	# 	print("Directory " , dirName ,  " already exists")    


def score(y_obs, y_predict, score_type='R2'):

	if score_type == 'R2':
	    score = r2_score(y_obs, y_predict)
	elif score_type == 'Pearson':
	    score = np.corrcoef(y_obs, y_predict)[0, 1]

	return score

def error(y_obs, y_predict, error_type='MAE'):
	if error_type == 'MAE':
		err = mean_absolute_error(y_obs, y_predict)
	elif error_type == 'RMSE':
	    err = np.sqrt(mean_squared_error(y_obs, y_predict))
	return err