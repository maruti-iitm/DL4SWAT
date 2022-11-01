# Pre-process the discharge and 20 SWAT parameter data for DL analysis
#   Water Year (WY) from 2014 to 2016
#   Oct-1-2013 to Sep-30-2016
# Pre-processing methods:
#	7 different methods and their impact on accuracy
#		1. StandardScaler
#		2. MinMaxScaler
#		3. MaxAbsScaler
#		4. RobustScaler
#		5. PowerTransformer (Yeo-Johnson) 
#		6. QuantileTransformer (uniform output)
#		7. QuantileTransformer (Gaussian output)
# Train/Val/Test splits:
#   Val = 10%, Test = 10%
#   Train --> 5%, 10%, 20%, 40%, 60%, and 80%
# SWAT Parameters and their indices:
#	'CN2',      --> 0
#	'ESCO',     --> 1
#	'RCHRG_DP', --> 2
#	'GWQMN',    --> 3
#	'GW_REVAP', --> 4
#	'REVAPMN',  --> 5
#	'GW_DELAY', --> 6
#	'ALPHA_BF', --> 7
#	'SOL_K',    --> 8
#	'SOL_AWC',  --> 9
#	'CH_N2',    --> 10
#	'CH_K2',    --> 11
#	'OV_N',     --> 12
#	'SFTMP',    --> 13
#	'SMTMP',    --> 14
#	'SMFMX',    --> 15
#	'TIMP',     --> 16
#	'EPCO',     --> 17
#	'PLAPS',    --> 18
#	'TLAPS'     --> 19
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import pickle
import time
import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
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

np.set_printoptions(precision=2)
print("sklearn version = ", sklearn.__version__)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#====================================================;
#  Function-1: Plot histogram of pre-processed data  ;
#====================================================;
def plot_histplot(data_list, num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos):

    #----------------------------------;
    #  Pre-processed data (histogram)  ;
    #----------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    #ax.set_xlim([0, 730])
    #ax.set_ylim([0, 80])
    ax.hist(data_list, bins = num_bins, label = label_name, \
    		edgecolor = 'k', alpha = 0.5, color = 'b', density = True)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    fig.savefig(fig_name + '.png')
    plt.close(fig)

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
np.random.seed(1337) #For reproducable results
#
path           = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat      = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_obs       = path + 'Data/Obs_Data/' #Obs data
path_ind       = path + 'Data/SWAT/Train_Val_Test_Indices/' #Train/Val/Test indices
path_pp_models = path + 'PreProcess_Models/' #Pre-processing models for standardization
path_pp_data   = path_swat + 'PreProcessed_Data/' #Pre-processed data
path_raw_data  = path_swat + 'Raw_Data/' #Raw data

#************************************;
#  2a. SWAT data for pre-processing  ;
#************************************;
df_p           = pd.read_csv(path_swat + 'swat_params_sobol_1000realz.csv', \
                        		index_col = 0) #[1000 rows x 20 columns]
df_q           = pd.read_csv(path_swat + 'flow_wy_sobol_1000realz.csv', \
                        		index_col = 0) #[1096 rows x 1001 columns] WY-2014 to WY-2016
df_q_obs       = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
                        		index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016
#
#['CN2', 'ESCO', 'RCHRG_DP', 'GWQMN', 'GW_REVAP', 'REVAPMN', 'GW_DELAY', 'ALPHA_BF', \
# 'SOL_K', 'SOL_AWC', 'CH_N2', 'CH_K2', 'OV_N', 'SFTMP', 'SMTMP',
# 'SMFMX', 'TIMP', 'EPCO', 'PLAPS', 'TLAPS'] #Total number of 20 parameters
#
swat_p_list    = df_p.columns.to_list() #SWAT params list -- A total of 20; length = 20
q_list         = df_q.columns.to_list()[1:] #Discharge realz list -- A total of 1000; length = 1000
#
x_p            = df_p.values #Matrix of values for SWAT params -- 1000 realz; (1000, 20)
x_q            = np.transpose(df_q.iloc[:,1:].values) #Matrix of values of discharge -- 1000 realz; (1000, 1096)
#
x_q_obs        = np.reshape(df_q_obs.values[:,0], (1,-1)) #Obs. raw discharge (1, 1096)
#
#mi_ind_list    = [14, 12, 8, 3, 11, 15, 7, 0, 16] #Order of sensitivities, a total of 9
#swat_p_mi_list = [swat_p_list[i] for i in mi_ind_list] #SWAT sensitive parameters a total of 9
#
print(np.argwhere(np.isnan(x_q)))

#*******************************************;
#  2b. np.log10 on non-negative parameters  ;
#*******************************************;
#p_min_list         = df_p.min(axis = 0).values #(20,)
#p_max_list         = df_p.max(axis = 0).values #(20,)
#nn_ind_list        = [1, 2, 3, 4, 5, 6, 7, 10, 11, 15, 16, 17, 18] #np.argwhere(p_min_list > 0)[:,0].tolist()
#x_p[:,nn_ind_list] = np.log10(x_p[:,nn_ind_list]) #np.log10(df_p.iloc[:,nn_ind_list]) #Log transform
##x_p[:,nn_ind_list] = np.power(10, x_p[:,nn_ind_list]) #Inverse log transform

#************************************************;
#  3. Develop train/val/test splits              ;
#     Val = 10%, Test = 10%					     ;
#     Train --> 5%, 10%, 20%, 40%, 60%, and 80%  ;
#************************************************;
num_realz  = 1000 #Sobol realz (total realz data)
num_train  = 800 #Training realz
num_val    = 100 #Validation realz
num_test   = 100 #Testing realz
#
train_index_list = np.genfromtxt(path_ind + "Train_Realz_" + str(num_train) + ".txt", \
									dtype = int, skip_header = 1)  
val_index_list   = np.genfromtxt(path_ind + "Val_Realz_" + str(num_val) + ".txt", \
									dtype = int, skip_header = 1)
test_index_list  = np.genfromtxt(path_ind + "Test_Realz_" + str(num_test) + ".txt", \
									dtype = int, skip_header = 1)
#
train_raw_q = x_q[train_index_list-1,:] #Raw training data (800, 1096)
val_raw_q   = x_q[val_index_list-1,:] #Raw val data (100, 1096)
test_raw_q  = x_q[test_index_list-1,:] #Raw test data (100, 1096)
#
train_raw_p = x_p[train_index_list-1,:] #Raw training data (800, 20)
val_raw_p   = x_p[val_index_list-1,:] #Raw val data (100, 20)
test_raw_p  = x_p[test_index_list-1,:] #Raw test data (100, 20)
#
np.save(path_raw_data + "train_raw_q_" + str(num_train) + ".npy", \
		train_raw_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_raw_data + "val_raw_q_" + str(num_val) + ".npy", \
		val_raw_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_raw_data + "test_raw_q_"  + str(num_test) + ".npy", \
		test_raw_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_raw_data + "train_raw_p_" + str(num_train) + ".npy", \
		train_raw_p) #Save train ground truth (800, 20) in *.npy file
np.save(path_raw_data + "val_raw_p_" + str(num_val) + ".npy", \
		val_raw_p) #Save val ground truth (100, 20) in *.npy file
np.save(path_raw_data + "test_raw_p_"  + str(num_test) + ".npy", \
		test_raw_p) #Save test ground truth (100, 20) in *.npy file

#*******************************************;
#  4. Pre-processing using Standard Scalar  ;
#*******************************************;
p_ss        = StandardScaler() #SWAT-params standard-scalar
q_ss        = StandardScaler() #SWAT-discharge standard-scalar
#
p_ss.fit(train_raw_p) #Fit standard-scalar for SWAT-params -- 800 realz (training)
q_ss.fit(train_raw_q) #Fit standard-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_ss_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_ss_" + str(num_train) + ".sav"
pickle.dump(p_ss, open(p_name, 'wb')) #Save the fitted standard-scalar (SS) SWAT-params model
pickle.dump(q_ss, open(q_name, 'wb')) #Save the fitted standard-scalar (SS) SWAT-discharge model
#
pp_ss       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params standard-scalar model
qq_ss       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge standard-scalar model
#
train_ss_p  = pp_ss.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_ss_p    = pp_ss.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_ss_p   = pp_ss.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_ss_q  = qq_ss.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_ss_q    = qq_ss.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_ss_q   = qq_ss.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_ss_q_" + str(num_train) + ".npy", \
		train_ss_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_ss_q_" + str(num_val) + ".npy", \
		val_ss_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_ss_q_"  + str(num_test) + ".npy", \
		test_ss_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_ss_p_" + str(num_train) + ".npy", \
		train_ss_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_ss_p_" + str(num_val) + ".npy", \
		val_ss_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_ss_p_"  + str(num_test) + ".npy", \
		test_ss_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_q_' + str(num_train)
plot_histplot(train_ss_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_ss_q_' + str(num_val)
plot_histplot(val_ss_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_ss_q_' + str(num_test)
plot_histplot(test_ss_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_SFTMP_' + str(num_train)
plot_histplot(train_ss_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ss_SFTMP_' + str(num_val)
plot_histplot(val_ss_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ss_SFTMP_' + str(num_test)
plot_histplot(test_ss_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_SFTMP_' + str(num_train)
plot_histplot(train_ss_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ss_SFTMP_' + str(num_val)
plot_histplot(val_ss_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ss_SFTMP_' + str(num_test)
plot_histplot(test_ss_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_ss_q_' + str(num_train)
plot_histplot(qq_ss.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#****************************************;
#  5. Pre-processing using MinMaxScaler  ;
#****************************************;
p_mms       = MinMaxScaler() #SWAT-params min-max-scalar
q_mms       = MinMaxScaler() #SWAT-discharge min-max-scalar
#
p_mms.fit(train_raw_p) #Fit min-max-scalar for SWAT-params -- 800 realz (training)
q_mms.fit(train_raw_q) #Fit min-max-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_mms_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_mms_" + str(num_train) + ".sav"
pickle.dump(p_mms, open(p_name, 'wb')) #Save the fitted min-max-scalar SWAT-params model
pickle.dump(q_mms, open(q_name, 'wb')) #Save the fitted min-max-scalar SWAT-discharge model
#
pp_mms       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params min-max-scalar model
qq_mms       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge min-max-scalar model
#
train_mms_p  = pp_mms.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_mms_p    = pp_mms.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_mms_p   = pp_mms.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_mms_q  = qq_mms.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_mms_q    = qq_mms.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_mms_q   = qq_mms.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_mms_q_" + str(num_train) + ".npy", \
		train_mms_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mms_q_" + str(num_val) + ".npy", \
		val_mms_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mms_q_"  + str(num_test) + ".npy", \
		test_mms_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_mms_p_" + str(num_train) + ".npy", \
		train_mms_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mms_p_" + str(num_val) + ".npy", \
		val_mms_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mms_p_"  + str(num_test) + ".npy", \
		test_mms_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mms_q_' + str(num_train)
plot_histplot(train_mms_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_mms_q_' + str(num_val)
plot_histplot(val_mms_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_mms_q_' + str(num_test)
plot_histplot(test_mms_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mms_SFTMP_' + str(num_train)
plot_histplot(train_mms_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mms_SFTMP_' + str(num_val)
plot_histplot(val_mms_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mms_SFTMP_' + str(num_test)
plot_histplot(test_mms_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mms_SFTMP_' + str(num_train)
plot_histplot(train_mms_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mms_SFTMP_' + str(num_val)
plot_histplot(val_mms_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mms_SFTMP_' + str(num_test)
plot_histplot(test_mms_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_mms_q_' + str(num_train)
plot_histplot(qq_mms.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#****************************************;
#  6. Pre-processing using MaxAbsScaler  ;
#****************************************;
p_mas       = MaxAbsScaler() #SWAT-params max-abs-scalar
q_mas       = MaxAbsScaler() #SWAT-discharge max-abs-scalar
#
p_mas.fit(train_raw_p) #Fit max-abs-scalar for SWAT-params -- 800 realz (training)
q_mas.fit(train_raw_q) #Fit max-abs-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_mas_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_mas_" + str(num_train) + ".sav"
pickle.dump(p_mas, open(p_name, 'wb')) #Save the fitted max-abs-scalar SWAT-params model
pickle.dump(q_mas, open(q_name, 'wb')) #Save the fitted max-abs-scalar SWAT-discharge model
#
pp_mas       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params max-abs-scalar model
qq_mas       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge max-abs-scalar model
#
train_mas_p  = pp_mas.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_mas_p    = pp_mas.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_mas_p   = pp_mas.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_mas_q  = qq_mas.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_mas_q    = qq_mas.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_mas_q   = qq_mas.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_mas_q_" + str(num_train) + ".npy", \
		train_mas_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mas_q_" + str(num_val) + ".npy", \
		val_mas_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mas_q_"  + str(num_test) + ".npy", \
		test_mas_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_mas_p_" + str(num_train) + ".npy", \
		train_mas_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mas_p_" + str(num_val) + ".npy", \
		val_mas_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mas_p_"  + str(num_test) + ".npy", \
		test_mas_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mas_q_' + str(num_train)
plot_histplot(train_mas_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_mas_q_' + str(num_val)
plot_histplot(val_mas_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_mas_q_' + str(num_test)
plot_histplot(test_mas_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mas_SFTMP_' + str(num_train)
plot_histplot(train_mas_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mas_SFTMP_' + str(num_val)
plot_histplot(val_mas_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mas_SFTMP_' + str(num_test)
plot_histplot(test_mas_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mas_SFTMP_' + str(num_train)
plot_histplot(train_mas_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mas_SFTMP_' + str(num_val)
plot_histplot(val_mas_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mas_SFTMP_' + str(num_test)
plot_histplot(test_mas_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_mas_q_' + str(num_train)
plot_histplot(qq_mas.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#****************************************;
#  7. Pre-processing using RobustScaler  ;
#****************************************;
p_rs        = RobustScaler(quantile_range = (25, 75)) #SWAT-params robust-scalar
q_rs        = RobustScaler(quantile_range = (25, 75)) #SWAT-discharge robust-scalar
#
p_rs.fit(train_raw_p) #Fit robust-scalar for SWAT-params -- 800 realz (training)
q_rs.fit(train_raw_q) #Fit robust-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_rs_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_rs_" + str(num_train) + ".sav"
pickle.dump(p_rs, open(p_name, 'wb')) #Save the fitted robust-scalar SWAT-params model
pickle.dump(q_rs, open(q_name, 'wb')) #Save the fitted robust-scalar SWAT-discharge model
#
pp_rs       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params robust-scalar model
qq_rs       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge robust-scalar model
#
train_rs_p  = pp_rs.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_rs_p    = pp_rs.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_rs_p   = pp_rs.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_rs_q  = qq_rs.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_rs_q    = qq_rs.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_rs_q   = qq_rs.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_rs_q_" + str(num_train) + ".npy", \
		train_rs_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_rs_q_" + str(num_val) + ".npy", \
		val_rs_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_rs_q_"  + str(num_test) + ".npy", \
		test_rs_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_rs_p_" + str(num_train) + ".npy", \
		train_rs_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_rs_p_" + str(num_val) + ".npy", \
		val_rs_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_rs_p_"  + str(num_test) + ".npy", \
		test_rs_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_rs_q_' + str(num_train)
plot_histplot(train_rs_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_rs_q_' + str(num_val)
plot_histplot(val_rs_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_rs_q_' + str(num_test)
plot_histplot(test_rs_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_rs_SFTMP_' + str(num_train)
plot_histplot(train_rs_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)'
fig_name    = path + 'Plots/Pre_Process/val_rs_SFTMP_' + str(num_val)
plot_histplot(val_rs_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_rs_SFTMP_' + str(num_test)
plot_histplot(test_rs_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_rs_SFTMP_' + str(num_train)
plot_histplot(train_rs_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_rs_SFTMP_' + str(num_val)
plot_histplot(val_rs_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_rs_SFTMP_' + str(num_test)
plot_histplot(test_rs_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_rs_q_' + str(num_train)
plot_histplot(qq_rs.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#**********************************************************;
#  8. Pre-processing using PowerTransformer (yeo-johnson)  ;
#**********************************************************;
p_ptyj      = PowerTransformer(method = "yeo-johnson") #SWAT-params power-transformer-yeo-johnson-scalar
q_ptyj      = PowerTransformer(method = "yeo-johnson") #SWAT-discharge power-transformer-yeo-johnson-scalar
#
p_ptyj.fit(train_raw_p) #Fit power-transformer-yeo-johnson-scalar for SWAT-params -- 800 realz (training)
q_ptyj.fit(train_raw_q) #Fit power-transformer-yeo-johnson-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_ptyj_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_ptyj_" + str(num_train) + ".sav"
pickle.dump(p_ptyj, open(p_name, 'wb')) #Save the fitted power-transformer-yeo-johnson-scalar SWAT-params model
pickle.dump(q_ptyj, open(q_name, 'wb')) #Save the fitted power-transformer-yeo-johnson-scalar SWAT-discharge model
#
pp_ptyj       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params power-transformer-yeo-johnson-scalar model
qq_ptyj       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge power-transformer-yeo-johnson-scalar model
#
train_ptyj_p  = pp_ptyj.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_ptyj_p    = pp_ptyj.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_ptyj_p   = pp_ptyj.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_ptyj_q  = qq_ptyj.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_ptyj_q    = qq_ptyj.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_ptyj_q   = qq_ptyj.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_ptyj_q_" + str(num_train) + ".npy", \
		train_ptyj_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_ptyj_q_" + str(num_val) + ".npy", \
		val_ptyj_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_ptyj_q_"  + str(num_test) + ".npy", \
		test_ptyj_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_ptyj_p_" + str(num_train) + ".npy", \
		train_ptyj_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_ptyj_p_" + str(num_val) + ".npy", \
		val_ptyj_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_ptyj_p_"  + str(num_test) + ".npy", \
		test_ptyj_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ptyj_q_' + str(num_train)
plot_histplot(train_ptyj_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_ptyj_q_' + str(num_val)
plot_histplot(val_ptyj_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_ptyj_q_' + str(num_test)
plot_histplot(test_ptyj_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ptyj_SFTMP_' + str(num_train)
plot_histplot(train_ptyj_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ptyj_SFTMP_' + str(num_val)
plot_histplot(val_ptyj_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ptyj_SFTMP_' + str(num_test)
plot_histplot(test_ptyj_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ptyj_SFTMP_' + str(num_train)
plot_histplot(train_ptyj_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ptyj_SFTMP_' + str(num_val)
plot_histplot(val_ptyj_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ptyj_SFTMP_' + str(num_test)
plot_histplot(test_ptyj_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_ptyj_q_' + str(num_train)
plot_histplot(qq_ptyj.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#*********************************************************;
#  9. Pre-processing using QuantileTransformer (uniform)  ;
#*********************************************************;
p_qtu       = QuantileTransformer(n_quantiles = num_train, output_distribution = "uniform") #SWAT-params quantile-transformer-uniform-scalar
q_qtu       = QuantileTransformer(n_quantiles = num_train, output_distribution = "uniform") #SWAT-discharge quantile-transformer-uniform-scalar
#
p_qtu.fit(train_raw_p) #Fit quantile-transformer-uniform-scalar for SWAT-params -- 800 realz (training)
q_qtu.fit(train_raw_q) #Fit quantile-transformer-uniform-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_qtu_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_qtu_" + str(num_train) + ".sav"
pickle.dump(p_qtu, open(p_name, 'wb')) #Save the fitted quantile-transformer-uniform-scalar SWAT-params model
pickle.dump(q_qtu, open(q_name, 'wb')) #Save the fitted quantile-transformer-uniform-scalar SWAT-discharge model
#
pp_qtu       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params quantile-transformer-uniform-scalar model
qq_qtu       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge quantile-transformer-uniform-scalar model
#
train_qtu_p  = pp_qtu.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_qtu_p    = pp_qtu.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_qtu_p   = pp_qtu.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_qtu_q  = qq_qtu.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_qtu_q    = qq_qtu.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_qtu_q   = qq_qtu.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_qtu_q_" + str(num_train) + ".npy", \
		train_qtu_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtu_q_" + str(num_val) + ".npy", \
		val_qtu_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtu_q_"  + str(num_test) + ".npy", \
		test_qtu_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_qtu_p_" + str(num_train) + ".npy", \
		train_qtu_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtu_p_" + str(num_val) + ".npy", \
		val_qtu_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtu_p_"  + str(num_test) + ".npy", \
		test_qtu_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtu_q_' + str(num_train)
plot_histplot(train_qtu_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_qtu_q_' + str(num_val)
plot_histplot(val_qtu_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_qtu_q_' + str(num_test)
plot_histplot(test_qtu_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtu_SFTMP_' + str(num_train)
plot_histplot(train_qtu_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)'
fig_name    = path + 'Plots/Pre_Process/val_qtu_SFTMP_' + str(num_val)
plot_histplot(val_qtu_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtu_SFTMP_' + str(num_test)
plot_histplot(test_qtu_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtu_SFTMP_' + str(num_train)
plot_histplot(train_qtu_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_qtu_SFTMP_' + str(num_val)
plot_histplot(val_qtu_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtu_SFTMP_' + str(num_test)
plot_histplot(test_qtu_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_qtu_q_' + str(num_train)
plot_histplot(qq_qtu.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#*********************************************************;
#  10. Pre-processing using QuantileTransformer (normal)  ;
#*********************************************************;
p_qtn       = QuantileTransformer(n_quantiles = num_train, output_distribution = "normal") #SWAT-params quantile-transformer-normal-scalar
q_qtn       = QuantileTransformer(n_quantiles = num_train, output_distribution = "normal") #SWAT-discharge quantile-transformer-normal-scalar
#
p_qtn.fit(train_raw_p) #Fit quantile-transformer-normal-scalar for SWAT-params -- 800 realz (training)
q_qtn.fit(train_raw_q) #Fit quantile-transformer-normal-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_qtn_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_qtn_" + str(num_train) + ".sav"
pickle.dump(p_qtn, open(p_name, 'wb')) #Save the fitted quantile-transformer-normal-scalar SWAT-params model
pickle.dump(q_qtn, open(q_name, 'wb')) #Save the fitted quantile-transformer-normal-scalar SWAT-discharge model
#
pp_qtn       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params quantile-transformer-normal-scalar model
qq_qtn       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge quantile-transformer-normal-scalar model
#
train_qtn_p  = pp_qtn.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_qtn_p    = pp_qtn.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_qtn_p   = pp_qtn.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_qtn_q  = qq_qtn.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_qtn_q    = qq_qtn.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_qtn_q   = qq_qtn.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_qtn_q_" + str(num_train) + ".npy", \
		train_qtn_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtn_q_" + str(num_val) + ".npy", \
		val_qtn_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtn_q_"  + str(num_test) + ".npy", \
		test_qtn_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_qtn_p_" + str(num_train) + ".npy", \
		train_qtn_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtn_p_" + str(num_val) + ".npy", \
		val_qtn_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtn_p_"  + str(num_test) + ".npy", \
		test_qtn_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtn_q_' + str(num_train)
plot_histplot(train_qtn_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_qtn_q_' + str(num_val)
plot_histplot(val_qtn_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_qtn_q_' + str(num_test)
plot_histplot(test_qtn_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtn_SFTMP_' + str(num_train)
plot_histplot(train_qtn_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)'
fig_name    = path + 'Plots/Pre_Process/val_qtn_SFTMP_' + str(num_val)
plot_histplot(val_qtn_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtn_SFTMP_' + str(num_test)
plot_histplot(test_qtn_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtn_SFTMP_' + str(num_train)
plot_histplot(train_qtn_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_qtn_SFTMP_' + str(num_val)
plot_histplot(val_qtn_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtn_SFTMP_' + str(num_test)
plot_histplot(test_qtn_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_qtn_q_' + str(num_train)
plot_histplot(qq_qtn.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)