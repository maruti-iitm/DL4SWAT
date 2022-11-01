# SWAT 20-parameter estimations from top-50 models (25000 and 50 parameter sets)
#    1. CNN-based inverse model to train/validate/test the proposed framework
#    3. INPUTS are 3 WY q-data (1096 days)
#       (a) Pre-processing based on StandardScaler for q-data
#    4. OUTPUTS are 20 SWAT params
#       (a) Pre-processing based on StandardScaler p-data
#    5. n_realz = 1000, Train/Val/Test = 80/10/10
#    6. 25000 parameters --> 50 mdoels * 100 realz * 5 obs data errors
#			Obs errors   --> 5%, 10%, 25%, 50%, 100%
#    7. Making sure that the DL estimated parameters are bounded and in physical range
#			DL_50_realz_v1.csv (physically meaningful and bounded) [50 rows x 20 columns]
#			DL_25000_realz_v1.csv (physically meaningful and bounded) [25000 rows x 20 columns]
#
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import time
import pandas as pd
import numpy as np
#
np.set_printoptions(precision=2)

#**************************************;
#  1. Set paths and import data files  ;
#**************************************;
path              = '/Users/mudu605/Desktop/Papers_PNNL/5_ML_YRB/AL/'
path_swat         = path + 'Data/SWAT/SWAT_v5/' #SWAT data
path_dl_models    = path + 'Trained_Models/v4/Top_50models_ss/' #Top-50 DL models using SWAT_v5 data
#
df_p_bounds       = pd.read_csv(path_swat + 'parameter_ranges_SWAT_20.csv', \
                                   index_col = 0) #[2 rows x 20 columns]
#
df_p              = pd.read_csv(path_swat + 'swat_params_sobol_1000realz.csv', \
                                   index_col = 0) #[1000 rows x 20 columns]
df_p_obs          = pd.read_csv(path_dl_models + "Top50DL_SWAT_obs_params.csv", \
									index_col = 0) #[50 rows x 20 columns]
df_p_obserr       = pd.read_csv(path_dl_models + "Top50DL_SWAT_obsdataerrors_params.csv", \
									index_col = 0) #[25000 rows x 20 columns]
#
swat_p_list       = df_p.columns.to_list() #SWAT params list -- A total of 20; length = 20
p_obs_ind_list    = df_p_obs.index.to_list() #50 -- Top-50 params without obs
p_obserr_ind_list = df_p_obserr.index.to_list() #25000 -- Top-50 params with obs errors
#
x_p_bounds        = df_p_bounds.values #(2,20)
x_p               = df_p.values #Matrix of values for SWAT params -- 1000 realz; (1000, 20)
x_p_obs           = df_p_obs.values #Matrix of values for Obs SWAT params -- 50 realz; (50, 20)
x_p_obserr        = df_p_obserr.values #Matrix of values for Obs + Err SWAT params -- 25000 realz; (25000, 20)
#
x_p_min           = np.min(df_p.values, axis = 0) #(20,)
x_p_max           = np.max(df_p.values, axis = 0) #(20,)
#
for i in range(0,20):
	print(i, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
			'{0:.3g}'.format(x_p_bounds[0,i]), '{0:.3g}'.format(x_p_bounds[1,i]))

#******************************************************;
#  2. Check bounds and update results (50 param sets)  ;
#******************************************************;
x_p_obs_new = np.zeros(x_p_obs.shape, dtype = float) #(50, 20)
#
for i in range(0,20): #Iterate over params
	for j in range(0,50): #Iterate over realz
		if not (x_p_obs[j,i] >= x_p_min[i] and x_p_obs[j,i] <= x_p_max[i]): #Not-within-bounds
			print(i, j, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
				x_p_obs[j,i])
			if x_p_obs[j,i] <= x_p_min[i]:
				x_p_obs_new[j,i] = x_p_min[i]
			else:
				x_p_obs_new[j,i] = x_p_max[i]
		else:
			x_p_obs_new[j,i] = x_p_obs[j,i]

for i in range(0,20): #Iterate over params
	for j in range(0,50): #Iterate over realz
		if not (x_p_obs[j,i] >= x_p_min[i] and x_p_obs[j,i] <= x_p_max[i]): #Not-within-bounds
			print(i, j, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
				x_p_obs[j,i], x_p_obs_new[j,i])

for i in range(0,20): #Iterate over params
	for j in range(0,50): #Iterate over realz
		if (x_p_obs[j,i] >= x_p_min[i] and x_p_obs[j,i] <= x_p_max[i]): #Within-bounds
			print(i, j, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
				x_p_obs[j,i], x_p_obs_new[j,i])	

df_p_obs_new = pd.DataFrame(x_p_obs_new, index = p_obs_ind_list, \
							columns = swat_p_list) #Create pd dataframe for obs SWAT params (updated) -- Top-50 models [50 rows x 20 columns]
#df_p_obs_new.to_csv(path_dl_models + "DL_50_realz_v1.csv")	#Save updated obs SWAT params [50 rows x 20 columns]

#*********************************************************;
#  3. Check bounds and update results (25000 param sets)  ;
#*********************************************************;
x_p_obserr_new = np.zeros(x_p_obserr.shape, dtype = float) #(25000, 20)
#
for i in range(0,20): #Iterate over params
	for j in range(0,25000): #Iterate over realz
		if not (x_p_obserr[j,i] >= x_p_min[i] and x_p_obserr[j,i] <= x_p_max[i]): #Not-within-bounds
			print(i, j, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
				x_p_obserr[j,i])
			if x_p_obserr[j,i] <= x_p_min[i]:
				x_p_obserr_new[j,i] = x_p_min[i]
			else:
				x_p_obserr_new[j,i] = x_p_max[i]
		else:
			x_p_obserr_new[j,i] = x_p_obserr[j,i]

for i in range(0,20): #Iterate over params
	for j in range(0,25000): #Iterate over realz
		if not (x_p_obserr[j,i] >= x_p_min[i] and x_p_obserr[j,i] <= x_p_max[i]): #Not-within-bounds
			print(i, j, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
				x_p_obserr[j,i], x_p_obserr_new[j,i])

for i in range(0,20): #Iterate over params
	for j in range(0,25000): #Iterate over realz
		if (x_p_obserr[j,i] >= x_p_min[i] and x_p_obserr[j,i] <= x_p_max[i]): #Within-bounds
			print(i, j, swat_p_list[i], '{0:.3g}'.format(x_p_min[i]), '{0:.3g}'.format(x_p_max[i]), \
				x_p_obserr[j,i], x_p_obserr_new[j,i])

df_p_obserr_new = pd.DataFrame(x_p_obserr_new, index = p_obserr_ind_list, \
							columns = swat_p_list) #Create pd dataframe for obserr SWAT params (updated) [25000 rows x 20 columns]
#df_p_obserr_new.to_csv(path_dl_models + "DL_25000_realz_v1.csv") #Save updated obserr SWAT params [25000 rows x 20 columns]