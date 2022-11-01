# Get the numbers of trained CNN-based inverse models (trained on NERSC)
# Total number of CNN models trained = 26250, need to see which are completed
#  Sanity check:
#	https://stackoverflow.com/questions/14798220/how-can-i-search-sub-folders-using-glob-glob-module
# AUTHOR: Maruti Kumar Mudunuru

import glob
import time
import re
import numpy as np

np.set_printoptions(precision=2)

#============================;
#  1. Start processing time  ;
#============================;
tic = time.perf_counter()

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path             = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/"
full_hph5_list   = list(range(1,26251))
hp_h5_list       = glob.glob(path + '1_InvCNNModel_ss/**/*.h5', recursive = True) #TF2 trained h5 files
print(len(hp_h5_list)) #(18750,)
#
train_model_list = np.array([int(re.findall(r'\d+', s)[4]) for s in hp_h5_list]) #(18750,)
train_model_list = np.sort(train_model_list) #(18750,)
#
nontrained_list  = np.array(list(set(full_hph5_list) - set(train_model_list.tolist()))) #from 18751 -- 26250
nontrained_list  = np.sort(nontrained_list) #7500

#=========================;
#  2.End processing time  ;
#=========================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)