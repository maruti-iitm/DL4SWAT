# Plot SWAT 20-parameter estimations from DL (25050) for various noise levels
#    1. DL set   (best 50 among 25050 realz)
#    2. 25000 parameters --> 50 mdoels * 100 realz * 5 obs data errors
#			Obs errors   --> 5%, 10%, 25%, 50%, 100%
#    3. Making sure that the DL estimated parameters are bounded and in physical range
#			DL_50_realz_v1.csv (physically meaningful and bounded) [50 rows x 20 columns]
#			DL_25000_realz_v1.csv (physically meaningful and bounded) [25000 rows x 20 columns]
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

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#================================;
#  Function-1: Box plots colors  ; 
#================================;
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color)
    plt.setp(bp['medians'], color = color)
    plt.setp(bp['whiskers'], color = color)
    plt.setp(bp['caps'], color = color)

#==============================================================;
#  Function-2: Box plots for 5 different observational errors  ; 
#==============================================================;
def plot_boxplot_params(x_pos, x_ticks_list, \
                             data1, data2, data3, data4, data5, data6, \
                             pos1, pos2, pos3, pos4, pos5, pos6, \
                             xmin, xmax, ymin, ymax, loc_pos, width_val, \
                             str_x_label, str_y_label, fig_name):

    #-------------------------------------;
    #  Boxplot for obs error params data  ;
    #-------------------------------------;
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
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16, length = 6, width = 2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #
    bp1 = ax.boxplot(data1, positions = pos1, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp2 = ax.boxplot(data2, positions = pos2, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp3 = ax.boxplot(data3, positions = pos3, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp4 = ax.boxplot(data4, positions = pos4, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp5 = ax.boxplot(data5, positions = pos5, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    bp6 = ax.boxplot(data6, positions = pos6, widths = width_val, vert = True, patch_artist = True, notch = True, showfliers = False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks_list)
    #
    set_box_color(bp1, color = 'black')
    set_box_color(bp2, color = 'black') 
    set_box_color(bp3, color = 'black') 
    set_box_color(bp4, color = 'black') 
    set_box_color(bp5, color = 'black')
    set_box_color(bp6, color = 'black')
    # 
    #colors = ['blue', 'green', 'red', ]
    colors = [[0,0,1,0.5],
              [0,1,0,0.5],
              [1,0,0,0.5]]
    for patch, color in zip(bp1['boxes'], [colors[0]]):
        patch.set_facecolor(color)
        #patch.set_edgecolor('m')
    for patch, color in zip(bp2['boxes'], [colors[0]]):
        patch.set_facecolor(color)
    for patch, color in zip(bp3['boxes'], [colors[0]]):
        patch.set_facecolor(color)
    for patch, color in zip(bp4['boxes'], [colors[0]]):
        patch.set_facecolor(color)
    for patch, color in zip(bp5['boxes'], [colors[0]]):
        patch.set_facecolor(color)
    for patch, color in zip(bp6['boxes'], [colors[0]]):
        patch.set_facecolor(color)
    hB,   = ax.plot([1],[1], color = colors[:][0], linewidth = 1.5)
    ax.legend((hB,), ('CNN',), loc = loc_pos)
    hB.set_visible(False)
    #tick_spacing = 10
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    #fig.savefig(fig_name + '.png', dpi = 1000) #Very high-res fig
    fig.savefig(fig_name + '.png', dpi = 300) #Medium res-fig
    plt.close(fig)

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path           = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat      = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_obs       = path + 'Data/Obs_Data/' #Obs data
#
params_list    = [r'CN2', r'ESCO', r'RCHRG\_DP', r'GWQMN', r'GW\_REVAP', \
					r'REVAPMN', r'GW\_DELAY', r'ALPHA\_BF', r'SOL\_K', r'SOL\_AWC', \
					r'CH\_N2', r'CH\_K2', r'OV\_N', r'SFTMP', r'SMTMP', \
					r'SMFMX', r'TIMP', r'EPCO', r'PLAPS', r'TLAPS'] #20
mi_ind_list    = [0, 2, 6, 7, 10, 11, 14, 15, 16, 17] #11 sensitive indices

#************************************************;
#  2a. Read the descending order sorted indices  ;
#************************************************;
df_argsm_glue_cp = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_glue_cp.csv", \
                                index_col = 0, dtype = int) #[1000 rows x 7 columns]
df_argsm_glue_vp = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_glue_vp.csv", \
                                index_col = 0, dtype = int) #[1000 rows x 7 columns]
df_argsm_dds_cp  = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dds_cp.csv", \
                                index_col = 0, dtype = int) #[5000 rows x 7 columns]
df_argsm_dds_vp  = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dds_vp.csv", \
                                index_col = 0, dtype = int) #[5000 rows x 7 columns]
df_argsm_dl_cp   = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dl_cp.csv", \
                                index_col = 0, dtype = int) #[25050 rows x 7 columns]
df_argsm_dl_vp   = pd.read_csv(path_swat + "DescendSort_DL_DDS_GLUE/argsm_dl_vp.csv", \
                                index_col = 0, dtype = int) #[25050 rows x 7 columns]

#***************************************************;
#  2b. Read the 25050 DL params (50 + 25000 realz)  ;
#***************************************************;
#df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/DL_50_realz_v1.csv', \
#                        index_col = 0) #[50 rows x 20 columns]
df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/Top50DL_SWAT_obs_params.csv', \
                        index_col = 0) #[50 rows x 20 columns]
print(np.argwhere(np.isnan(df_p_full.values)))
#
x_p_dl1      = df_p_full.values #Matrix of values -- 50 realz; (50, 20)
#
#df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/DL_25000_realz_v1.csv', \
#                        index_col = 0) #[25000 rows x 20 columns]
df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/Top50DL_SWAT_obsdataerrors_params.csv', \
                        index_col = 0) #[25000 rows x 20 columns]
print(np.argwhere(np.isnan(df_p_full.values)))
#
x_p_dl2      = df_p_full.values #Matrix of values -- 25000 realz; (25000, 20)
#
temp_p_dl    = np.concatenate((x_p_dl1, x_p_dl2)) #(25050, 20)
#
x_p_dltop50  = copy.deepcopy(temp_p_dl[df_argsm_dl_cp.iloc[0:50,4].values,:]) #(50, 20)
x_p_dl       = copy.deepcopy(temp_p_dl) #(25050, 20)

#***************************;
#  3a. SFTMP boxplots (DL)  ;
#***************************;
i = 13 #SFTMP
#
print('Min and Max for SFTMP-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #0.316 0.983
print('Min and Max for SFTMP-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #-1.26 2.15
print('Min and Max for SFTMP-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #-1.18 1.66
print('Min and Max for SFTMP-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #-1.93 1.48
print('Min and Max for SFTMP-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #-1.86 1.71
print('Min and Max for SFTMP-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #-1.71 2.14
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = -3
ymax         = 3
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = 'SFTMP'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_SFTMP_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***************************;
#  3b. CH_K2 boxplots (DL)  ;
#***************************;
i = 11 #CH_K2
#
print('Min and Max for CH_K2-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #175 200
print('Min and Max for CH_K2-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #141 200
print('Min and Max for CH_K2-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #136 200
print('Min and Max for CH_K2-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #141 200
print('Min and Max for CH_K2-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #126 200
print('Min and Max for CH_K2-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #136 200
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 120
ymax         = 200
loc_pos      = 'lower left'
str_x_label  = 'Relative observational error'
str_y_label  = r'CH\_K2'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_CH_K2_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#******************************;
#  3c. ALPHA_BF boxplots (DL)  ;
#******************************;
i = 7 #ALPHA_BF
#
print('Min and Max for ALPHA_BF-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #0.421 0.599
print('Min and Max for ALPHA_BF-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #0.184 0.825
print('Min and Max for ALPHA_BF-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #0.208 0.766
print('Min and Max for ALPHA_BF-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #0.207 0.781
print('Min and Max for ALPHA_BF-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #0.232 0.671
print('Min and Max for ALPHA_BF-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #0.253 0.697
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 0.1
ymax         = 0.9
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = r'ALPHA\_BF'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_ALPHA_BF_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#******************************;
#  3d. RCHRG_DP boxplots (DL)  ;
#******************************;
i = 2 #RCHRG_DP
#
print('Min and Max for RCHRG_DP-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #0.0046 0.343
print('Min and Max for RCHRG_DP-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #0.000488 0.63
print('Min and Max for RCHRG_DP-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #0.000488 0.684
print('Min and Max for RCHRG_DP-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #0.000488 0.614
print('Min and Max for RCHRG_DP-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #0.15 0.556
print('Min and Max for RCHRG_DP-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #0.0144 0.585
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 0.0
ymax         = 0.7
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = r'RCHRG\_DP'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_RCHRG_DP_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***************************;
#  3e. CH_N2 boxplots (DL)  ;
#***************************;
i = 10 #CH_N2
#
print('Min and Max for CH_N2-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #0.0883 0.123
print('Min and Max for CH_N2-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #0.0709 0.15
print('Min and Max for CH_N2-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #0.0754 0.15
print('Min and Max for CH_N2-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #0.0694 0.139
print('Min and Max for CH_N2-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #0.0709 0.136
print('Min and Max for CH_N2-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #0.054 0.141
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 0.0
ymax         = 0.2
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = r'CH\_N2'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_CH_N2_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***************************;
#  3f. SMTMP boxplots (DL)  ;
#***************************;
i = 14 #SMTMP
#
print('Min and Max for SMTMP-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #3.54 5
print('Min and Max for SMTMP-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #0.753 5
print('Min and Max for SMTMP-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #0.544 5
print('Min and Max for SMTMP-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #0.449 5
print('Min and Max for SMTMP-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #0.325 5
print('Min and Max for SMTMP-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #-1.11 5
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = -1.5
ymax         = 5
loc_pos      = 'lower left'
str_x_label  = 'Relative observational error'
str_y_label  = r'SMTMP'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_SMTMP_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***************************;
#  3g. TIMP boxplots (DL)  ;
#***************************;
i = 16 #TIMP
#
print('Min and Max for TIMP-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #0.435 1
print('Min and Max for TIMP-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #0.0105 1
print('Min and Max for TIMP-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #0.0105 1
print('Min and Max for TIMP-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #0.0105 1
print('Min and Max for TIMP-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #0.232 0.741
print('Min and Max for TIMP-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #0.145 0.765
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 0
ymax         = 1
loc_pos      = 'lower left'
str_x_label  = 'Relative observational error'
str_y_label  = r'TIMP'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_TIMP_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#*************************;
#  3h. CN2 boxplots (DL)  ;
#*************************;
i = 0 #CN2
#
print('Min and Max for CN2-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #-0.0473 0.264
print('Min and Max for CN2-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #-0.137 0.296
print('Min and Max for CN2-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #-0.0913 0.3
print('Min and Max for CN2-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #-0.137 0.3
print('Min and Max for CN2-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #-0.0702 0.264
print('Min and Max for CN2-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #-0.123 0.3
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = -0.15
ymax         = 0.2
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = r'CN2'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_CN2_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***************************;
#  3i. SMFMX boxplots (DL)  ;
#***************************;
i = 15 #SMFMX
#
print('Min and Max for SMFMX-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #3.7 5.41
print('Min and Max for SMFMX-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #1.65 6.9
print('Min and Max for SMFMX-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #1.4 6.9
print('Min and Max for SMFMX-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #1.4 6.9
print('Min and Max for SMFMX-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #2.04 4.93
print('Min and Max for SMFMX-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #2.27 5.27
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 1
ymax         = 7
loc_pos      = 'lower left'
str_x_label  = 'Relative observational error'
str_y_label  = r'SMFMX'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_SMFMX_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#******************************;
#  3j. GW_DELAY boxplots (DL)  ;
#******************************;
i = 6 #GW_DELAY
#
print('Min and Max for GW_DELAY-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #1.05 43.9
print('Min and Max for GW_DELAY-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #1.05 79.9
print('Min and Max for GW_DELAY-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #1.05 81.7
print('Min and Max for GW_DELAY-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #1.05 84.5
print('Min and Max for GW_DELAY-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #1.05 48.5
print('Min and Max for GW_DELAY-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #5.89 57.5
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 1
ymax         = 60
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = r'GW\_DELAY'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_GW_DELAY_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#**************************;
#  3k. EPCO boxplots (DL)  ;
#**************************;
i = 17 #EPCO
#
print('Min and Max for EPCO-CNN (0%) = ', '{0:.3g}'.format(np.min(x_p_dltop50[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dltop50[:,i]))) #0.0105 0.712
print('Min and Max for EPCO-CNN (5%) = ', '{0:.3g}'.format(np.min(x_p_dl[50:5050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[50:5050,i]))) #0.0105 1
print('Min and Max for EPCO-CNN (10%) = ', '{0:.3g}'.format(np.min(x_p_dl[5050:10050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[5050:10050,i]))) #0.0105 1
print('Min and Max for EPCO-CNN (25%) = ', '{0:.3g}'.format(np.min(x_p_dl[10050:15050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[10050:15050,i]))) #0.0105 1
print('Min and Max for EPCO-CNN (50%) = ', '{0:.3g}'.format(np.min(x_p_dl[15050:20050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[15050:20050,i]))) #0.223 0.587
print('Min and Max for EPCO-CNN (100%) = ', '{0:.3g}'.format(np.min(x_p_dl[20050:25050,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[20050:25050,i]))) #0.209 0.624
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
x_ticks_list = [r'0\%', r'5\%', r'10\%', r'25\%', r'50\%', r'100\%']
xmin         = 0
xmax         = 25
ymin         = 0
ymax         = 1
loc_pos      = 'upper left'
str_x_label  = 'Relative observational error'
str_y_label  = r'EPCO'
fig_name     = path + 'Plots/Noisy_DLSets/' + str(i) + '_EPCO_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5]
pos4         = [14.5]
pos5         = [18.5]     
pos6         = [22.5]  
#
test1        = [x_p_dltop50[:,i]] #DL
test2        = [x_p_dl[50:5050,i]] #DL
test3        = [x_p_dl[5050:10050,i]] #DL
test4        = [x_p_dl[10050:15050,i]] #DL
test5        = [x_p_dl[15050:20050,i]] #DL
test6        = [x_p_dl[20050:25050,i]] #DL
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, test4, test5, test6, \
                    pos1, pos2, pos3, pos4, pos5, pos6, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)