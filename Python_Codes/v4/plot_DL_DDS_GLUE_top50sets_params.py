# SWAT 20-parameter estimations from top-50 models of DL, DDS, and GLUE
#    1. DL set   (best 50 among 25050 realz)
#    2. DDS set  (best 50 among 5000 realz)
#    3. GLUE set (best 50 among 1000 realz)
#    4. 25000 parameters --> 50 mdoels * 100 realz * 5 obs data errors
#			Obs errors   --> 5%, 10%, 25%, 50%, 100%
#    5. Making sure that the DL estimated parameters are bounded and in physical range
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

#====================================================================;
#  Function-2: Box plots for sensitive params (DL vs. DDS vs. GLUE)  ; 
#====================================================================;
def plot_boxplot_params(x_pos, x_ticks_list, \
                             data1, data2, data3, \
                             pos1, pos2, pos3, \
                             xmin, xmax, ymin, ymax, loc_pos, width_val, \
                             str_x_label, str_y_label, fig_name):

    #-----------------------------------------------;
    #  Boxplot for params data  (DL vs DDS vs GLUE) ;
    #-----------------------------------------------;
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
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks_list)
    #
    set_box_color(bp1, color = 'black')
    set_box_color(bp2, color = 'black') 
    set_box_color(bp3, color = 'black') 
    # 
    #colors = ['blue', 'green', 'red']
    colors = [[0,0,1,0.5],
              [0,1,0,0.5],
              [1,0,0,0.5]]
    for patch, color in zip(bp1['boxes'], [colors[0]]):
        patch.set_facecolor(color)
        #patch.set_edgecolor('m')
        #print(patch, color)
    for patch, color in zip(bp2['boxes'], [colors[1]]):
        patch.set_facecolor(color)
    for patch, color in zip(bp3['boxes'], [colors[2]]):
        patch.set_facecolor(color)
    hB,   = ax.plot([1],[1], color = colors[:][0], linewidth = 1.5)
    hG,   = ax.plot([1],[1], color = colors[:][1], linewidth = 1.5)
    hR,   = ax.plot([1],[1], color = colors[:][2], linewidth = 1.5)
    ax.legend((hB, hG, hR), ('CNN', 'DDS', 'GLUE'), loc = loc_pos)
    hR.set_visible(False)
    hG.set_visible(False)
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

#*****************************************;
#  2b. Read the GLUE params (1000 realz)  ;
#*****************************************;
df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/GLUE_1000_realz.csv', \
                        index_col = 0) #[1000 rows x 20 columns]
print(np.argwhere(np.isnan(df_p_full.values)))
#
temp_p_glue  = df_p_full.values #Matrix of values -- 5000 realz; (5000, 20)
#
x_p_glue     = copy.deepcopy(temp_p_glue[df_argsm_glue_cp.iloc[0:50,4].values,:]) #(50, 20)

#****************************************;
#  2c. Read the DDS params (5000 realz)  ;
#****************************************;
df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/DDS_5000_realz.csv', \
                        index_col = None).iloc[:,0:20] #[5000 rows x 20 columns]
print(np.argwhere(np.isnan(df_p_full.values)))
#
temp_p_dds   = df_p_full.values #Matrix of values -- 5000 realz; (5000, 20)
#
x_p_dds      = copy.deepcopy(temp_p_dds[df_argsm_dds_cp.iloc[0:50,4].values,:]) #(50, 20)

#***************************************************;
#  2d. Read the 25050 DL params (50 + 25000 realz)  ;
#***************************************************;
df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/DL_50_realz_v1.csv', \
                        index_col = 0) #[50 rows x 20 columns]
print(np.argwhere(np.isnan(df_p_full.values)))
#
x_p_dl1      = df_p_full.values #Matrix of values -- 50 realz; (50, 20)
#
df_p_full    = pd.read_csv(path_swat + 'ParamSets_DL_DDS_GLUE/DL_25000_realz_v1.csv', \
                        index_col = 0) #[25000 rows x 20 columns]
print(np.argwhere(np.isnan(df_p_full.values)))
#
x_p_dl2      = df_p_full.values #Matrix of values -- 25000 realz; (25000, 20)
#
temp_p_dl    = np.concatenate((x_p_dl1, x_p_dl2)) #(25050, 20)
#
x_p_dl       = copy.deepcopy(temp_p_dl[df_argsm_dl_cp.iloc[0:50,4].values,:]) #(50, 20)

#********************************************;
#  3a. SFTMP boxplots (DL vs. DDS vs. GLUE)  ;
#********************************************;
i = 13 #SFTMP
#
print('Min and Max for SFTMP-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #-1.34 3.27
print('Min and Max for SFTMP-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #0.71 2.08
print('Min and Max for SFTMP-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #0.316 0.983
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = -1.5
ymax         = 3.5
loc_pos      = 'upper left'
str_x_label  = 'Calibration method'
str_y_label  = 'SFTMP'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_SFTMP_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#********************************************;
#  3b. CH_K2 boxplots (DL vs. DDS vs. GLUE)  ;
#********************************************;
i = 11 #CH_K2
#
print('Min and Max for CH_K2-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #69.7 197
print('Min and Max for CH_K2-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #132 167
print('Min and Max for CH_K2-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #175 200
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 65
ymax         = 210
loc_pos      = 'lower left'
str_x_label  = 'Calibration method'
str_y_label  = r'CH\_K2'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_CH_K2_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***********************************************;
#  3c. ALPHA_BF boxplots (DL vs. DDS vs. GLUE)  ;
#***********************************************;
i = 7 #ALPHA_BF
#
print('Min and Max for ALPHA_BF-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #0.0669 0.863
print('Min and Max for ALPHA_BF-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #0.315 0.613
print('Min and Max for ALPHA_BF-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #0.421 0.599
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 0
ymax         = 1
loc_pos      = 'lower left'
str_x_label  = 'Calibration method'
str_y_label  = r'ALPHA\_BF'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_ALPHA_BF_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***********************************************;
#  3d. RCHRG_DP boxplots (DL vs. DDS vs. GLUE)  ;
#***********************************************;
i = 2 #RCHRG_DP
#
print('Min and Max for RCHRG_DP-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #0.0112 0.99
print('Min and Max for RCHRG_DP-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #0.00235 0.0575
print('Min and Max for RCHRG_DP-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #0.0046 0.343
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 0
ymax         = 1
loc_pos      = 'upper left'
str_x_label  = 'Calibration method'
str_y_label  = r'RCHRG\_DP'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_RCHRG_DP_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#********************************************;
#  3e. CH_N2 boxplots (DL vs. DDS vs. GLUE)  ;
#********************************************;
i = 10 #CH_N2
#
print('Min and Max for CH_N2-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #0.0349 0.15
print('Min and Max for CH_N2-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #0.05 0.132
print('Min and Max for CH_N2-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #0.0883 0.123
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 0
ymax         = 0.2
loc_pos      = 'upper left'
str_x_label  = 'Calibration method'
str_y_label  = r'CH\_N2'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_CH_N2_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#********************************************;
#  3f. SMTMP boxplots (DL vs. DDS vs. GLUE)  ;
#********************************************;
i = 14 #SMTMP
#
print('Min and Max for SMTMP-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #-4.39 4.72
print('Min and Max for SMTMP-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #-0.277 3.96
print('Min and Max for SMTMP-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #3.54 5
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = -5
ymax         = 5
loc_pos      = 'lower left'
str_x_label  = 'Calibration method'
str_y_label  = r'SMTMP'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_SMTMP_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#*******************************************;
#  3g. TIMP boxplots (DL vs. DDS vs. GLUE)  ;
#*******************************************;
i = 16 #TIMP
#
print('Min and Max for TIMP-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #0.0153 0.807
print('Min and Max for TIMP-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #0.0157 0.712
print('Min and Max for TIMP-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #0.435 1
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 0
ymax         = 1
loc_pos      = 'lower left'
str_x_label  = 'Calibration method'
str_y_label  = r'TIMP'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_TIMP_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#*******************************************;
#  3h. CN2 boxplots (DL vs. DDS vs. GLUE)  ;
#*******************************************;
i = 0 #CN2
#
print('Min and Max for CN2-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #-0.295 0.294
print('Min and Max for CN2-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #-0.299 -0.229
print('Min and Max for CN2-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #-0.0473 0.264
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = -0.3
ymax         = 0.3
loc_pos      = 'lower left'
str_x_label  = 'Calibration method'
str_y_label  = r'CN2'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_CN2_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#********************************************;
#  3i. SMFMX boxplots (DL vs. DDS vs. GLUE)  ;
#********************************************;
i = 15 #SMFMX
#
print('Min and Max for SMFMX-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #1.41 6.66
print('Min and Max for SMFMX-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #2.33 5.01
print('Min and Max for SMFMX-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #3.7 5.41
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 1
ymax         = 7
loc_pos      = 'lower left'
str_x_label  = 'Calibration method'
str_y_label  = r'SMFMX'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_SMFMX_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#***********************************************;
#  3j. GW_DELAY boxplots (DL vs. DDS vs. GLUE)  ;
#***********************************************;
i = 6 #GW_DELAY
#
print('Min and Max for GW_DELAY-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #1.15 99
print('Min and Max for GW_DELAY-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #12.8 71.2
print('Min and Max for GW_DELAY-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #1.05 43.9
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 1
ymax         = 100
loc_pos      = 'upper left'
str_x_label  = 'Calibration method'
str_y_label  = r'GW\_DELAY'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_GW_DELAY_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)

#*******************************************;
#  3k. EPCO boxplots (DL vs. DDS vs. GLUE)  ;
#*******************************************;
i = 17 #EPCO
#
print('Min and Max for EPCO-GLUE = ', '{0:.3g}'.format(np.min(x_p_glue[:,i])), \
        '{0:.3g}'.format(np.max(x_p_glue[:,i]))) #0.0192 0.993
print('Min and Max for EPCO-DDS = ', '{0:.3g}'.format(np.min(x_p_dds[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dds[:,i]))) #0.188 0.946
print('Min and Max for EPCO-CNN = ', '{0:.3g}'.format(np.min(x_p_dl[:,i])), \
        '{0:.3g}'.format(np.max(x_p_dl[:,i]))) #0.0105 0.712
#
width_val    = 0.5
x_pos        = [2.5, 6.5, 10.5]
x_ticks_list = [r'CNN', r'DDS', r'GLUE']
xmin         = 0
xmax         = 11
ymin         = 0
ymax         = 1
loc_pos      = 'upper left'
str_x_label  = 'Calibration method'
str_y_label  = r'EPCO'
fig_name     = path + 'Plots/DL_DDS_GLUE_Sets/' + str(i) + '_EPCO_GLUE_DDS_CNN_cp'
#
pos1         = [2.5]
pos2         = [6.5]
pos3         = [10.5] 
#
test1        = [x_p_dl[:,i]] #DL
test2        = [x_p_dds[:,i]] #DDS
test3        = [x_p_glue[:,i]] #GLUE
#
plot_boxplot_params(x_pos, x_ticks_list, \
                    test1, test2, test3, \
                    pos1, pos2, pos3, \
                    xmin, xmax, ymin, ymax, loc_pos, width_val, \
                    str_x_label, str_y_label, fig_name)