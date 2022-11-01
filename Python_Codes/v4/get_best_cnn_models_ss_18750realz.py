# Get and sort the best models (Top-10, 100, and 1000) models out of 26250
#   Sorting based on the following performance metrics
#		1. Validation MSE
#		2. MSE of top-9 sensitive parameters
#		3. R2-score of top-9 sensitive parameters
# In SS all models are trained
# Total number of CNN models trained = 26250
# AUTHOR: Maruti Kumar Mudunuru

import glob
import re
from functools import reduce
import operator
#
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
from sklearn.metrics import mean_squared_error, r2_score
#
import hydroeval as he
import spotpy #A Statistical Parameter Optimization Tool for Python

np.set_printoptions(precision=2)
print("sklearn version = ", sklearn.__version__)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#**************************************;
#  1. Set paths and import data files  ;
#**************************************;
#path           = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT/"
path             = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/"
path_swat      = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_obs       = path + 'Data/Obs_Data/' #Obs data
path_ind       = path + 'Data/SWAT/Train_Val_Test_Indices/' #Train/Val/Test indices
path_pp_models = path + 'PreProcess_Models/' #Pre-processing models for standardization
path_pp_data   = path_swat + 'PreProcessed_Data/' #Pre-processed data
path_raw_data  = path_swat + 'Raw_Data/' #Raw data
#
num_realz      = 1000 #Sobol realz (total realz data)
num_train      = 800 #Training realz
num_val        = 100 #Validation realz
num_test       = 100 #Testing realz
#
df_p           = pd.read_csv(path_swat + 'swat_params_sobol_1000realz.csv', \
                                   index_col = 0) #[1000 rows x 20 columns]
swat_p_list    = df_p.columns.to_list() #SWAT params list -- A total of 20; length = 20
mi_ind_list    = [13, 11, 7, 2, 10, 14, 16, 0, 15] #Order of sensitivities, a total of 9
#mi_ind_list    = [13, 11] #Order of sensitivities, a total of 9
swat_p_mi_list = [swat_p_list[i] for i in mi_ind_list] #SWAT sensitive parameters a total of 9
#
test_raw_p     = np.load(path_raw_data + "test_raw_p_" + \
                            str(num_test) + ".npy") #Save test ground truth (100, 20) in *.npy file
#
sclr_name      = "ss" #Standard Scaler
p_name         = path_pp_models + "p_" + sclr_name + "_" + str(num_train) + ".sav" #SWAT pre-processor
pp_scalar      = pickle.load(open(p_name, 'rb')) #Load already created SWAT pre-processing model
#
test_p         = np.load(path_pp_data + "test_" + sclr_name + "_p_"  + \
                            str(num_test) + ".npy") #Load ss p-data pre-processed (100, 20) in *.npy file
test_p_it      = pp_scalar.inverse_transform(test_p) #IT of test swat-p data using ss
#
print(r2_score(test_raw_p,test_p_it)) #Effect of pytj transformation is nill as r2-score = 1.0

#******************************************************************************;
#  2. Read the trained data, model, and inferences to get performance metrics  ;
#*******************************************************************************;
full_hph5_list   = np.arange(1, 18751)
#hp_h5_list       = np.delete(full_hph5_list, obj = [119, 25, 115, 89]) #(132,) #[120, 26, 116, 90] realz numbers should be deleted 
hp_h5_list       = copy.deepcopy(full_hph5_list)
metrics_list     = np.zeros((len(hp_h5_list), 6), dtype = float) #(132,6) [id, num_epochs, train_loss, val_loss, r2_test_mi, mse_test_mi]
#metrics_list     = np.zeros((137, 6), dtype = float)
#
#for i in range(1,12):
for i in hp_h5_list:
       path_fl_sav = path + "1_InvCNNModel_ss/" + str(i) + "_model/" #i-th hp-dl-model folder
       #print(path_fl_sav)
       #
       train_csv  = path_fl_sav + "InvCNNModel_Loss.csv"
       df_hist    = pd.read_csv(train_csv)
       num_epochs = int(df_hist.values[-1,0])
       train_loss = df_hist.values[-1,1]
       val_loss   = df_hist.values[-1,2]
       #
       test_pred_p = np.load(path_fl_sav + "test_pred_p_" + \
                            str(num_test) + ".npy") #Load normalized test ss predictions (100, 20)
       #
       r2_test_mi  = r2_score(test_p[:,mi_ind_list],test_pred_p[:,mi_ind_list])
       mse_test_mi = mean_squared_error(test_p[:,mi_ind_list],test_pred_p[:,mi_ind_list])
       #
       metrics_list[i-1,:] = i, num_epochs, train_loss, val_loss, r2_test_mi, mse_test_mi
       print(i, num_epochs, train_loss, val_loss, r2_test_mi, mse_test_mi)

an_array         = copy.deepcopy(metrics_list)
vl_sorted_array  = copy.deepcopy(an_array[np.argsort(an_array[:,3])]) #Sort by validation loss
#vl_sorted_array  = copy.deepcopy(an_array[np.argsort(an_array[:,2])]) #Sort by validation loss
r2_sorted_array  = copy.deepcopy(an_array[np.argsort(an_array[:,4])[::-1]]) #Sort by r2_test_mi
mse_sorted_array = copy.deepcopy(an_array[np.argsort(an_array[:,5])])  #Sort by mse_test_mi
#
#np.save(path + "DL_Metrics_ss/unsorted_metrics.npy", an_array) #(26249,6) [id, num_epochs, train_loss, val_loss, r2_test_mi, mse_test_mi]
#np.save(path + "DL_Metrics_ss/vl_sorted_metrics.npy", vl_sorted_array) #(26249,6)
#np.save(path + "DL_Metrics_ss/r2_sorted_metrics.npy", r2_sorted_array) #(26249,6)
#np.save(path + "DL_Metrics_ss/mse_sorted_metrics.npy", mse_sorted_array) #(26249,6)
#
cols_list = ['i', 'num_epochs', 'train_loss', 'val_loss', 'r2_test_mi', 'mse_test_mi']
df_metrics = pd.DataFrame(an_array, columns = cols_list)
df_vl      = pd.DataFrame(vl_sorted_array, columns = cols_list)
df_r2      = pd.DataFrame(r2_sorted_array, columns = cols_list)
df_mse     = pd.DataFrame(mse_sorted_array, columns = cols_list)

#*************************************************************************;
#  3. Read the best trained model -- SWAT params of obs data (50 models)  ;
#*************************************************************************;
temp_list       =  reduce(operator.concat, [vl_sorted_array[1:26,0].tolist(), \
                            r2_sorted_array[0:24,0].tolist(), \
                            mse_sorted_array[1:25,0].tolist()])
best_model_list = np.unique(np.array(temp_list, dtype = int))
num_best_models = len(best_model_list)
best_obs_p_list = np.zeros((num_best_models,len(swat_p_list)), dtype = float) #(50,20)
#
for i in range(0,num_best_models):
       path_fl_sav        = path + "1_InvCNNModel_ss/" + str(best_model_list[i]) + "_model/" #i-th hp-dl-model folder
       print(path_fl_sav)
       obs_p_csv          = path_fl_sav + "SWAT_obs_params.csv"
       df_obs_p           = pd.read_csv(obs_p_csv, index_col = 0, header = 0)
       best_obs_p_list[i] = copy.deepcopy(df_obs_p.values[0,:])

best_obs_p_list[:,3] = np.abs(best_obs_p_list[:,3])
#
index_list  = ['InvCNNModel-' + str(counter) for counter in best_model_list]
df_bestp    = pd.DataFrame(best_obs_p_list, index = index_list, \
                            columns = swat_p_list) #Create pd dataframe for obs SWAT params -- Top-50 models
df_bestp_mi = df_bestp.iloc[:,mi_ind_list]
#df_bestp.to_csv(path + "Top50DL_SWAT_obs_params.csv")
#
for i in range(0,20):
       print(i, swat_p_list[i], np.min(df_p.iloc[:,i]), np.max(df_p.iloc[:,i]), \
              np.min(df_bestp.iloc[:,i]), np.max(df_bestp.iloc[:,i]))

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)