# SWAT 20-parameter estimations from top-50 models (noisy test data realz predictions)
#    1. CNN-based inverse model to train/validate/test the proposed framework
#    3. INPUTS are 3 WY q-data (1096 days)
#       (a) Pre-processing based on StandardScaler for q-data
#    4. OUTPUTS are 20 SWAT params
#       (a) Pre-processing based on StandardScaler p-data
#    5. n_realz = 10000, Noisy test realz (5%, 10%, 25%, 50%, and 100%)
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
bm_all_list = [46, 51, 52, 53, 56, 57, 58, 59, 196, \
			201, 203, 204, 206, 208, 356, 357, 358, \
			507, 657, 807, 3801, 3806, 3807, 3808, 5136, \
			7740, 8281, 8737, 8886, 8899, 8905, 8909, 9659, \
			9936, 10398, 12359, 13092, 13097, 13248, 13253, 13403, \
			13409, 13555, 14009, 14748, 15503, 15651, 16402, 17007, 17896] #Top-50 DL model folder names
bm_list    = [59]
#
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
	path_raw_data = path_swat + 'Raw_Data/' #Raw data

	#---------------------------------------------------;
	#  2. Load obs data with errors and pre-processors  ;
	#---------------------------------------------------;
	noise_list     = [0.05, 0.1, 0.25, 0.5, 1.0] #Noise value list
	noise          = 1.0 #Noise value of 5%, 10%, 25%, 50%, and 100%
	#
	num_train      = 800 #Training realz
	num_val        = 100 #Validation realz
	num_test       = 100 #Testing realz
	#
	num_realz      = 10000 #Test realz with noise
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
	test_raw_q     = np.load(path_raw_data + "test_raw_q_100.npy") #Raw test-q data (100, 1096) -- Without noise
	test_raw_p     = np.load(path_raw_data + "test_raw_p_100.npy") #Raw test-p data (100, 20)   -- Without noise
	#
	x_q_noisy      = np.load(path_swat + "TestErrorData_All/" + "test_raw_q_" + str(noise) + "_noise.npy") #(10000, 1096)

	#-------------------------------------------------;
	#  3. Load Top-50 DL models and make predictions  ;
	#-------------------------------------------------;
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
		xt_q_noisy    = qq_scalar.transform(x_q_noisy) #scaled test q-data with noise
		x_pred_p      = inv_model.predict(xt_q_noisy) #CNN-predictions
		x_pred_p_it   = pp_scalar.inverse_transform(x_pred_p) #Inverse transform CNN-predictions
		x_pred_p_it[:,[2,5,6]] = copy.deepcopy(np.abs(x_pred_p_it[:,[2,5,6]]))
		df_xn_p       = pd.DataFrame(x_pred_p_it, \
										index = ['InvCNNModel-' + str(i) + '-' + str(j) for j in range(1,10001)], \
										columns = swat_p_list) #Create pd dataframe for obs SWAT params
		df_xn_p.to_csv(path_swat + "TestErrorData_All/SWAT_test_params_" + str(noise) + "_noise.csv") #[10000 rows x 20 columns]
		#
		print('\n\n\n') #End of 20-param model

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)