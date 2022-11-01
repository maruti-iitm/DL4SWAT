#SWAT 20-parameter estimations from top-50 models (25000 parameter sets)
#    1. CNN-based inverse model to train/validate/test the proposed framework
#    3. INPUTS are 3 WY q-data (1096 days)
#       (a) Pre-processing based on StandardScaler for q-data
#    4. OUTPUTS are 20 SWAT params
#       (a) Pre-processing based on StandardScaler p-data
#    5. n_realz = 1000, Train/Val/Test = 80/10/10
#    6. 25000 parameters --> 50 mdoels * 100 realz * 5 obs data errors
#
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import time
import yaml
import pydot
import graphviz
import pydotplus
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
import sklearn #'1.0.2'
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
#
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import CSVLogger

#*************************************;
#  0. Top-50 DL model folder numbers  ;
#*************************************;
bm_list = [46, 51, 52, 53, 56, 57, 58, 59, 196, \
			201, 203, 204, 206, 208, 356, 357, 358, \
			507, 657, 807, 3801, 3806, 3807, 3808, 5136, \
			7740, 8281, 8737, 8886, 8899, 8905, 8909, 9659, \
			9936, 10398, 12359, 13092, 13097, 13248, 13253, 13403, \
			13409, 13555, 14009, 14748, 15503, 15651, 16402, 17007, 17896] #Top-50 DL model folder names

if __name__ == '__main__':

	#=========================;
	#  Start processing time  ;
	#=========================;
	tic = time.perf_counter()

	#--------------------------------------------;
	#  1. Set paths and top-50 DL model numbers  ;
	#--------------------------------------------;
	path            = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
	path_swat       = path + 'Data/SWAT/SWAT_v5/' #SWAT data
	path_pp_models  = path + 'PreProcess_Models/' #Pre-processing models for standardization
	path_dl_models  = path + 'Trained_Models/v4/Top_50models_ss/' #Top-50 DL models using SWAT_v5 data

	#---------------------------------------------------;
	#  2. Load obs data with errors and pre-processors  ;
	#---------------------------------------------------;
	num_realz      = 1000 #Sobol realz (total realz data)
	num_train      = 800 #Training realz
	num_val        = 100 #Validation realz
	num_test       = 100 #Testing realz
	#
	sclr_name      = "ss" #Standard Scaler
	p_name         = path_pp_models + "p_" + sclr_name + "_" + str(num_train) + ".sav" #SWAT pre-processor
	q_name         = path_pp_models + "q_" + sclr_name + "_" + str(num_train) + ".sav" #q-data pre-processor
	#
	pp_scalar      = pickle.load(open(p_name, 'rb')) #Load already created SWAT pre-processing model
	qq_scalar      = pickle.load(open(q_name, 'rb')) #Load already created q-data pre-processing model
	#
	df_p           = pd.read_csv(path_swat + 'swat_params_sobol_1000realz.csv', \
									index_col = 0) #[1000 rows x 20 columns]
	swat_p_list    = df_p.columns.to_list() #SWAT params list -- A total of 20; length = 20
	#
	x_q_obs_noisy  = np.load(path_swat + 'Errors_Obs_Data/' + \
							'obs_flow_wy_data_errors.npy') #Obs data with errors (500, 1096) in *.npy file
	
	#-------------------------------------------------;
	#  3. Load Top-50 DL models and make predictions  ;
	#-------------------------------------------------;
	appended_data = []
	#
	K.clear_session()
	#
	for i in bm_list:
		path_fl_sav =  path_dl_models + str(i) + "_model/"
		print(i, path_fl_sav)
		#  
		inv_model = tf.keras.models.load_model(path_fl_sav + "Inv_CNN_Model") #Load saved 20-parameter inverse model
		inv_model.summary() #Print baseline model summary (CNN, StandardScalar)
		print(inv_model.layers[-1].output_shape[1:]) #Print output shape (no. of neurons in out-layer)
		#
		xt_q_obs      = qq_scalar.transform(x_q_obs_noisy) #scaled obs. q-data with noise
		obs_pred_p    = inv_model.predict(xt_q_obs) #CNN-predictions
		obs_pred_p_it = pp_scalar.inverse_transform(obs_pred_p) #Inverse transform CNN-predictions
		obs_pred_p_it[:,[2,5,6]] = copy.deepcopy(np.abs(obs_pred_p_it[:,[2,5,6]]))
		df_obs_p      = pd.DataFrame(obs_pred_p_it, \
										index = ['InvCNNModel-' + str(i) + '-' + str(j) for j in range(1,501)], \
										columns = swat_p_list) #Create pd dataframe for obs SWAT params
		df_obs_p.to_csv(path_fl_sav + "SWAT_obsdataerrors_params.csv") #[500 rows x 20 columns]
		#
		print('\n\n\n') #End of 20-param model
		#
		appended_data.append(df_obs_p) #store DataFrame in list

	df_temp      = pd.concat(appended_data) # see pd.concat documentation for more info
	df_obs_p_all = df_temp.copy(deep=True)
	df_obs_p_all.to_csv(path_dl_models + "Top50DL_SWAT_obsdataerrors_params.csv") #[25000 rows x 20 columns]

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)