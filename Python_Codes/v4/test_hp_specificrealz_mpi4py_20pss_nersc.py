# Test how to run realization on different cores (LAPTOP, PINKLADY, and NERSC)
#  (SWAT_v5 -- 1000 sobol sequence with only one soil layer)
#
# module load texlive
# module load python/3.9-anaconda-2021.11
# module load cray-hdf5-parallel
# conda activate /global/common/software/m1800/maruti_python/conda/myenv1
# export HDF5_USE_FILE_LOCKING=FALSE
#
# srun -n 32 -c 2 --cpu-bind=cores python test_hp_specificrealz_mpi4py_20pss_nersc.py #Haswell
# srun -n 68 -c 4 --cpu-bind=cores python test_hp_specificrealz_mpi4py_20pss_nersc.py #KNL
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

from mpi4py import MPI

#====================================================================;
#  Function-1: Train individual models (mpi4py calls this function)  ; 
#====================================================================;
def get_trained_models(k_child_rank, start_at_this_hpfolder):

    #-------------------;
    #  0. Get realz_id  ;
    #-------------------;
    #print(k_child_rank, start_at_this_hpfolder, k_child_rank+start_at_this_hpfolder)
    #
    counter       = k_child_rank + start_at_this_hpfolder - 1
    path_testing  = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/"
    path_fl_sav   =  path_testing + "1_InvCNNModel_ss/" + str(counter) + "_model/" #i-th model folder
    print(path_fl_sav + 'hp_input_deck.txt')

#********************************************************************;
#  Set paths, load preprocessed encoded PCA dataand dump .csv files  ;
#********************************************************************;
#
if __name__ == '__main__':

    #============================;
    #  1. Start processing time  ;
    #============================;
    tic = time.perf_counter()

    #=======================================;
    #  2. MPI communicator, size, and rank  ;
    #=======================================;
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #============================================;
    #  3. Number of realz per process/rank/core  ;
    #============================================;
    num_total = 1 #Total number of realization that needs to run on a given process/rank/core
    #realz_id_list = [[40, 50, 660], [71, 82, 90], [100, 110, 120]]

    #=========================================;
    #  4. MPI send and receive realz numbers  ;
    #=========================================;
    if rank == 0:
        for i in range(size-1,-1,-1):
            realz_id = [0]*num_total  #Realization list
            #
            for j in range(num_total):
                #print(j + num_total*i + 1, realz_id)
                realz_id[j] = j + num_total*i + 1
            #print('rank and realz_id = ', rank, realz_id)
            #
            if i > 0: 
                comm.send(realz_id, dest = i)
                #comm.send(realz_id_list[i-1], dest = i) #You can also send specific realz list to child process
    else:
        realz_id = comm.recv(source = 0)
        #print('rank, realz_id = ', rank, realz_id)

    #==========================================;
    #  5. Run DL model training for each realz ;
    #==========================================;
    for k in realz_id:
        start_at_this_hpfolder = 1 #1, 18751, 21751, 24001, 25501
        get_trained_models(k, start_at_this_hpfolder)
        print('rank, k, realz_id = ', rank, k, realz_id, k+start_at_this_hpfolder-1)

    #=========================;
    #  6.End processing time  ;
    #=========================;
    toc = time.perf_counter()
    print('Time elapsed in seconds = ', toc - tic)