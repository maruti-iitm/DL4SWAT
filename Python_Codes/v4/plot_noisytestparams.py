# Plot noisy test data params (all 100 test realz)
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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
import sklearn #'1.0.2'
from sklearn.metrics import mean_squared_error, r2_score

np.set_printoptions(precision=2)
print("sklearn version = ", sklearn.__version__)

#================================;
#  Function-1: Box plots colors  ; 
#================================;
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color)
    plt.setp(bp['medians'], color = color)
    plt.setp(bp['whiskers'], color = color)
    plt.setp(bp['caps'], color = color)

#=============================================================;
#  Function-1: Plot params + confidence interval (all realz)  ;
#=============================================================;
def plot_err_data(t_list, data_matrix, y_true, y_mean_b, y_lb, y_ub, \
                  str_x_label, str_y_label, fig_name, \
                  legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax):

    #---------------------;
    #  Params error data  ;
    #---------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure(figsize=(20,5))
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 18)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'x', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.set_xlim([0, 101])
    ax.set_ylim([ymin, ymax])
    colors = [[1,0,0,0.5],
              [0,1,0,0.5],
              [0,0,1,0.5]]
    sns.lineplot(t_list, y_mean_b, linestyle = '', markersize = 8, marker = 'o', color = 'b', \
                 label = obs_label_list[0]) #CNN
    sns.lineplot(t_list, y_true, linestyle = '', markersize = 8, marker = 'X', color = 'k', \
                 label = obs_label_list[1]) #True. data
    #ax.fill_between(t_list, y_lb, y_ub, linestyle = 'solid', linewidth = 0.5, \
    #                color = 'bisque', alpha = alpha_val) #Mean +/ 2*std or 95% CI
    bp1 = ax.boxplot(data_matrix, positions = t_list, widths = 0.05, vert = True, patch_artist = True, notch = True, showfliers = False)
    set_box_color(bp1, color = 'b')
    x_tick_spacing = x_num_ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
    ax.set_xticklabels([-20, 0, 20, 40, 60, 80, 100])
    y_tick_spacing = y_num_ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc = legend_loc)
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000)
    plt.close(fig)

#------------------------------------;
#  1. Set paths and test error data  ;
#------------------------------------;
path            = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat       = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_pp_models  = path + 'PreProcess_Models/' #Pre-processing models for standardization
path_raw_data   = path_swat + 'Raw_Data/' #Raw data
#
params_list    = [r'CN2', r'ESCO', r'RCHRG\_DP', r'GWQMN', r'GW\_REVAP', \
					r'REVAPMN', r'GW\_DELAY', r'ALPHA\_BF', r'SOL\_K', r'SOL\_AWC', \
					r'CH\_N2', r'CH\_K2', r'OV\_N', r'SFTMP', r'SMTMP', \
					r'SMFMX', r'TIMP', r'EPCO', r'PLAPS', r'TLAPS'] #20
#
noise_list     = [0.05, 0.1, 0.25, 0.5, 1.0] #Noise value list
noise          = 1.0 #Noise value of 5%, 10%, 25%, 50%, and 100%
#
x_p            = np.load(path_raw_data + "test_raw_p_100.npy") #Raw test-p data (100, 20)   -- Without noise
df_xn_p        = pd.read_csv(path_swat + "TestErrorData_All/SWAT_test_params_" + \
								str(noise) + "_noise.csv", index_col = 0) #[10000 rows x 20 columns]
xn_p           = df_xn_p.values #(10000, 20)

#-------------------------------;
#  2. Get the mean, lb, and ub  ;
#-------------------------------;
counter         = 0
#
mean_err_data   = np.zeros((100,20))
lb_err_data     = np.zeros((100,20))
ub_err_data     = np.zeros((100,20))
#
for i in range(0,100):
    mean_err_data[i,:] = np.mean(xn_p[counter:counter+100,:], axis = 0)
    lb_err_data[i,:]   = np.min(xn_p[counter:counter+100,:], axis = 0)
    ub_err_data[i,:]   = np.max(xn_p[counter:counter+100,:], axis = 0)
    #
    counter            = counter + 100

#-------------------------------;
#  3a. SFTMP plot (index = 13)  ;
#-------------------------------;
index    = 13
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = 'SFTMP (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_SFTMP_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 1
ymin           = -5.75
ymax           = 5.75
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#-------------------------------;
#  3b. CH_K2 plot (index = 11)  ;
#-------------------------------;
index    = 11
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'CH\_K2 (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_CH_K2_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 20
ymin           = 0
ymax           = 200
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#---------------------------------;
#  3c. ALPHA_BF plot (index = 7)  ;
#---------------------------------;
index    = 7
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'ALPHA\_BF (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_ALPHA_BF_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 0.2
ymin           = 0
ymax           = 1
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#---------------------------------;
#  3d. RCHRG_DP plot (index = 2)  ;
#---------------------------------;
index    = 2
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'RCHRG\_DP (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_RCHRG_DP_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 0.2
ymin           = 0
ymax           = 1
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#-------------------------------;
#  3e. CH_N2 plot (index = 10)  ;
#-------------------------------;
index    = 10
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'CH\_N2 (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_CH_N2_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 0.05
ymin           = 0
ymax           = 0.15
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#-------------------------------;
#  3f. SMTMP plot (index = 14)  ;
#-------------------------------;
index    = 14
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'SMTMP (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_SMTMP_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 1
ymin           = -5.75
ymax           = 5.75
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#------------------------------;
#  3g. TIMP plot (index = 16)  ;
#------------------------------;
index    = 16
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'TIMP (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_TIMP_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 0.2
ymin           = 0
ymax           = 1
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#----------------------------;
#  3h. CN2 plot (index = 0)  ;
#----------------------------;
index    = 0
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'CN2 (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_CN2_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 0.1
ymin           = -0.3
ymax           = 0.3
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#-------------------------------;
#  3i. SMFMX plot (index = 15)  ;
#-------------------------------;
index    = 15
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'SMFMX (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_SMFMX_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 1
ymin           = 1.4
ymax           = 7.0
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#---------------------------------;
#  3j. GW_DELAY plot (index = 6)  ;
#---------------------------------;
index    = 6
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'GW\_DELAY (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_GW_DELAY_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 20
ymin           = 1
ymax           = 100
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)

#------------------------------;
#  3k. EPCO plot (index = 17)  ;
#------------------------------;
index    = 17
data_arr = []
counter  = 0
#
for i in range(0,100):
	temp = copy.deepcopy(xn_p[counter:counter+100,index])
	data_arr.append(temp.flatten())
	# 
	counter = counter + 100
#
t_list         = np.arange(0,100)
str_x_label    = 'Test realizations'
str_y_label    = r'EPCO (prior range)'
fig_name       = path + 'Plots/Noisy_TestSets/' + str(index) + '_EPCO_vs_Allrealz' 
legend_loc     = 'upper left'
obs_label_list = ['CNN', 'Synthetic truth']
x_num_ticks    = 20
y_num_ticks    = 0.2
ymin           = 0
ymax           = 1
#
plot_err_data(t_list, data_arr, x_p[:,index], mean_err_data[:,index], \
				lb_err_data[:,index], ub_err_data[:,index], \
				str_x_label, str_y_label, fig_name, \
				legend_loc, obs_label_list, x_num_ticks, y_num_ticks, ymin, ymax)