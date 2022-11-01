# get_inv_cnn_models_ss_nersc.py to estimate all 20-SWAT parameters (NERSC)
#    1. CNN-based inverse model to train/validate/test the proposed framework
#    3. INPUTS are 3 WY q-data (1096 days)
#       (a) Pre-processing based on StandardScaler for q-data
#    4. OUTPUTS are 20 SWAT params
#       (a) Pre-processing based on StandardScaler p-data
#    5. n_realz = 1000, Train/Val/Test = 80/10/10
#
# mpirun -n 50 python get_inv_cnn_models_20pss_nersc.py >> hp_output_ss_nersc.txt
# srun -n 32 -c 2 --cpu-bind=cores python get_inv_cnn_models_20pss_nersc.py >> hp_output_ss_nersc.txt #Haswell
# srun -n 68 -c 4 --cpu-bind=cores python get_inv_cnn_models_20pss_nersc.py >> hp_output_ss_nersc.txt #KNL
# killall python
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

#=====================================================================;
#  Function-1: Full Inverse Model (for all hyperparameter scenarios)  ;             
#=====================================================================;
def get_cnn_model(nq_comps, n_features, nswat_comps, \
					filter_sizes, kernel_sizes, pool_sizes, \
					num_layers, dropout_value):

	#----------------------------------------------;
	#  Construct 1D CNN with multiple Conv layers  ;
	#  inputs = nq_comps (features)         	   ;
	#  outputs = nswat_comps (SWAT params)         ;
	#----------------------------------------------;
	q_shape    = (nq_comps, n_features) #(n_timesteps, n_features) = (1096,1)
	#
	input_layer = Input(shape = q_shape, name = "discharge_data") #Input layer
	x           = input_layer
	#
	for i in range(0,num_layers):
		filter_size = filter_sizes[i]
		kernel_size = kernel_sizes[i]
		pool_size   = pool_sizes[i]
		#
		x = Conv1D(filters      = filter_size, \
					kernel_size = kernel_size, \
					activation  = 'relu', \
					name        = 'Conv-' + str(i+1))(x)
		x = MaxPool1D(pool_size = pool_size, \
						name    = 'MaxPooling-' + str(i+1))(x)

	x     = Flatten(name = 'Flatten-1')(x)
	x     = Dropout(rate = dropout_value)(x)
	
	out   = Dense(nswat_comps, name = 'Out-Dense-1')(x)
	model = Model(input_layer, out, name = "CNN-Inv-Model")

	return model

#=========================================================;
#  Function-2: Plot training and validation loss ('mse')  ; 
#=========================================================;
def plot_tv_loss(hist, epochs, path_fl_sav):

	#---------------------;
	#  Plot loss ('mse')  ;
	#---------------------;
	legend_properties = {'weight':'bold'}
	fig               = plt.figure()
	#
	plt.rc('text', usetex = True)
	plt.rcParams['font.family']     = ['sans-serif']
	plt.rcParams['font.sans-serif'] = ['Lucida Grande']
	plt.rc('legend', fontsize = 14)
	ax = fig.add_subplot(111)
	ax.set_xlabel('Epoch', fontsize = 24, fontweight = 'bold')
	ax.set_ylabel('Loss (MSE)', fontsize = 24, fontweight = 'bold')
	#plt.grid(True)
	ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
	ax.set_xlim([0, epochs])
	#ax.set_ylim([0.35, 1])
	e_list = [i for i in range(0,epochs)]
	sns.lineplot(e_list, hist['loss'], linestyle = 'solid', linewidth = 1.5, \
					color = 'b', label = 'Training') #Training loss
	sns.lineplot(e_list, hist['val_loss'], linestyle = 'solid', linewidth = 1.5, \
					color = 'm', label = 'Validation') #Validation loss
	#tick_spacing = 100
	#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	ax.legend(loc = 'upper right')
	fig.tight_layout()
	fig.savefig(path_fl_sav + 'Loss.pdf')
	fig.savefig(path_fl_sav + 'Loss.png')
	plt.close(fig)

#=======================================================================;
#  Function-3: Plot one-to-one for train/val/test (All 20 SWAT params)  ; 
#=======================================================================;
def plot_gt_pred(x, y, param_id, fl_name, str_x_label, str_y_label):

	#------------------------------------------------;
	#  Plot one-to-one (ground truth vs. predicted)  ;
	#------------------------------------------------;
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
	min_val = np.min(x)
	max_val = np.max(x)
	ax.set_xlim([min_val, max_val])
	ax.set_ylim([min_val, max_val])
	sns.lineplot([min_val, max_val], [min_val, max_val], \
					linestyle = 'solid', linewidth = 1.5, \
					color = 'r') #One-to-One line
	sns.scatterplot(x, y, color = 'b', marker = 'o')
	#tick_spacing = 100
	#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	#ax.legend(loc = 'upper right')
	ax.set_aspect(1./ax.get_data_ratio())
	fig.tight_layout()
	#fig.savefig(fl_name + str(param_id) + '.pdf')
	fig.savefig(fl_name + str(param_id) + '.png')
	plt.close(fig)

#====================================================================;
#  Function-4: Train individual models (mpi4py calls this function)  ; 
#====================================================================;
def get_trained_models(k_child_rank, start_at_this_hpfolder):

	#-------------------;
	#  0. Get realz_id  ;
	#-------------------;
	counter = k_child_rank + start_at_this_hpfolder - 1
	#
	#counter_list  = [1, 18751, 21751, 24001, 25501]
	#counter       = counter_list[0]

	#------------------------------------------------;
	#  1. Get pre-processed data (all realizations)  ;
	#------------------------------------------------;
	#path = os.getcwd() #Get current directory path
	path           = '/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/'
	path_swat      = path + 'Data/SWAT/SWAT_v5/' #SWAT data
	path_obs       = path + 'Data/Obs_Data/' #Obs data
	path_ind       = path + 'Data/SWAT/Train_Val_Test_Indices/' #Train/Val/Test indices
	path_pp_models = path + 'PreProcess_Models/' #Pre-processing models for standardization
	path_pp_data   = path_swat + 'PreProcessed_Data/' #Pre-processed data
	path_raw_data  = path_swat + 'Raw_Data/' #Raw data
	#
	path_testing   = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/" #26250 models and their inputs are here
	#
	path_fl_sav    =  path_testing + "1_InvCNNModel_ss/" + str(counter) + "_model/" #i-th hp-dl-model folder
	print(path_fl_sav + 'hp_input_deck.txt')

	#--------------------------------;
	#  2. Get other initializations  ;
	#--------------------------------;
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
	#mi_ind_list    = [14, 12, 8, 3, 11, 15, 7, 0, 16] #Order of sensitivities, a total of 9
	#swat_p_mi_list = [swat_p_list[i] for i in mi_ind_list] #SWAT sensitive parameters a total of 9
	#
	df_q_obs       = pd.read_csv(path_obs + 'obs_flow_wy_data.csv', \
									index_col = 0) #[1096 rows x 1 columns] WY-2014 to WY-2016
	x_q_obs        = np.reshape(df_q_obs.values[:,0], (1,-1)) #Obs. raw discharge (1, 1096)
	#
	#p_min_list         = df_p.min(axis = 0).values #(20,)
	#p_max_list         = df_p.max(axis = 0).values #(20,)
	#nn_ind_list        = [1, 2, 3, 4, 5, 6, 7, 10, 11, 15, 16, 17, 18] #np.argwhere(p_min_list > 0)[:,0].tolist()

	#-------------------------------------------------------------------;
	#  3. Load train/val/test *.npy data -- One of the 7-preprocessors  ;
	#-------------------------------------------------------------------; 
	train_q = np.load(path_pp_data + "train_" + sclr_name + "_q_" + \
						str(num_train) + ".npy") #Train q-data pre-processed (800, 1096) in *.npy file
	val_q   = np.load(path_pp_data + "val_" + sclr_name + "_q_" + \
						str(num_val) + ".npy") #Save q-data pre-processed (100, 1096) in *.npy file
	test_q  = np.load(path_pp_data + "test_" + sclr_name + "_q_"  + \
						str(num_test) + ".npy") #Save q-data pre-processed (100, 1096) in *.npy file 
	#
	train_p = np.load(path_pp_data + "train_" + sclr_name + "_p_" + \
						str(num_train) + ".npy") #Train p-data pre-processed (800, 20) in *.npy file
	val_p   = np.load(path_pp_data + "val_" + sclr_name + "_p_" + \
						str(num_val) + ".npy") #Save p-data pre-processed (100, 20) in *.npy file
	test_p  = np.load(path_pp_data + "test_" + sclr_name + "_p_"  + \
						str(num_test) + ".npy") #Save p-data pre-processed (100, 20) in *.npy file

	#----------------------------------------------;
	#  4a. Model training initialization           ;
	#      (read from a hyperparameter .txt file)  ;
	#----------------------------------------------;
	K.clear_session()
	#
	np.random.seed(1337)
	#
	nq_comps      = train_q.shape[1] #1096
	nswat_comps   = train_p.shape[1] #20
	n_features    = 1 #Number of channels
	#
	fl_id         = open(path_fl_sav + 'hp_input_deck.txt') #Read the hp .txt file line by line
	#
	hp_line_list  = fl_id.readlines()
	#
	num_layers    = int(hp_line_list[0].strip().split(" = ", 1)[1])
	filter_sizes  = [int(i) for i in hp_line_list[1].strip().split(" = ", 1)[1].split(",")]
	kernel_sizes  = [int(i) for i in hp_line_list[2].strip().split(" = ", 1)[1].split(",")]
	pool_sizes    = [int(i) for i in hp_line_list[3].strip().split(" = ", 1)[1].split(",")]
	dropout_value = float(hp_line_list[4].strip().split(" = ", 1)[1])
	lr_values     = float(hp_line_list[5].strip().split(" = ", 1)[1])
	epochs        = int(hp_line_list[6].strip().split(" = ", 1)[1])
	batch_size    = int(hp_line_list[7].strip().split(" = ", 1)[1])
	#
	for hp_line in hp_line_list:
		temp = hp_line.strip().split(" = ", 1)
		print(temp, len(temp))
	fl_id.close()
	
	#-------------------------------------;
	#  4b. Model training and validation  ;
	#-------------------------------------; 
	inv_model = get_cnn_model(nq_comps, n_features, nswat_comps, \
								filter_sizes, kernel_sizes, pool_sizes, \
								num_layers, dropout_value) #Inverse CNN model
	inv_model.summary() #Model summary
	tf.keras.utils.plot_model(inv_model, path_fl_sav + "Full-Inv-CNN-Model-a.png", show_shapes = True)
	tf.keras.utils.plot_model(inv_model, path_fl_sav + "Full-Inv-CNN-Model-b.png", show_shapes = False)
	#
	opt        = Adam(learning_rate = lr_values) #Optimizer and learning rate
	loss       = "mse" #MSE loss function
	inv_model.compile(opt, loss = loss)
	train_csv  = path_fl_sav + "InvCNNModel_Loss.csv"
	csv_logger = CSVLogger(train_csv)
	callbacks  = [csv_logger] 
    #
	history    = inv_model.fit(x = train_q, y = train_p, \
								epochs = epochs, batch_size = batch_size, \
								validation_data = (val_q, val_p), \
								verbose = 2, callbacks = callbacks)
	hist = history.history
	print("Done training")
	#print(hist.keys())
	time.sleep(60)

	#--------------------------------------;
	#  5. Plot train and val loss ('mse')  ;
	#     (loss and epoch stats)           ;
	#--------------------------------------;
	plot_tv_loss(hist, epochs, path_fl_sav)
	#
	df_hist        = pd.read_csv(train_csv)
	val_f1         = df_hist['val_loss']
	min_val_f1     = val_f1.min()
	min_val_f1_df  = df_hist[val_f1 == min_val_f1]
	min_epochs     = min_val_f1_df['epoch'].values
	min_val_loss   = min_val_f1_df['val_loss'].values
	min_train_loss = min_val_f1_df['loss'].values
	#
	print(min_val_f1_df)

	#----------------------------------------;
	#  6. Model prediction (train/val/test)  ;
	#----------------------------------------;
	train_pred_p = inv_model.predict(train_q)
	val_pred_p   = inv_model.predict(val_q)
	test_pred_p  = inv_model.predict(test_q)
	#
	np.save(path_fl_sav + "train_pred_p_" + str(num_train) + ".npy", \
		train_pred_p) #Save train ground truth (800, 20) in *.npy file (normalized)
	np.save(path_fl_sav + "val_pred_p_" + str(num_val) + ".npy", \
		val_pred_p) #Save val ground truth (100, 20) in *.npy file (normalized)
	np.save(path_fl_sav + "test_pred_p_" + str(num_test) + ".npy", \
		test_pred_p) #Save test ground truth (100, 20) in *.npy file (normalized)

	#------------------------------------------------------------;
	#  7a. Inverse transformed SWAT parameters (train/val/test)  ;
	#------------------------------------------------------------;
	train_p_it      = pp_scalar.inverse_transform(train_p)
	val_p_it        = pp_scalar.inverse_transform(val_p)
	test_p_it       = pp_scalar.inverse_transform(test_p)
	#
	train_pred_p_it = pp_scalar.inverse_transform(train_pred_p)
	val_pred_p_it   = pp_scalar.inverse_transform(val_pred_p)
	test_pred_p_it  = pp_scalar.inverse_transform(test_pred_p)
	#
	np.save(path_fl_sav + "train_pred_p_it" + str(num_train) + ".npy", \
		train_pred_p_it) #Save train ground truth (800, 20) in *.npy file
	np.save(path_fl_sav + "val_pred_p_it" + str(num_val) + ".npy", \
		val_pred_p_it) #Save val ground truth (100, 20) in *.npy file
	np.save(path_fl_sav + "test_pred_p_it" + str(num_test) + ".npy", \
		test_pred_p_it) #Save test ground truth (100, 20) in *.npy file

	#-------------------------------------------------------------------;
	#  7b. Inverse log10-transform of SWAT parameters (train/val/test)  ;
	#-------------------------------------------------------------------;
	#train_p_it[:,nn_ind_list]      = np.power(10, train_p_it[:,nn_ind_list])
	#val_p_it[:,nn_ind_list]        = np.power(10, val_p_it[:,nn_ind_list])
	#test_p_it[:,nn_ind_list]       = np.power(10, test_p_it[:,nn_ind_list])
	#
	#train_pred_p_it[:,nn_ind_list] = np.power(10, train_pred_p_it[:,nn_ind_list])
	#val_pred_p_it[:,nn_ind_list]   = np.power(10, val_pred_p_it[:,nn_ind_list])
	#test_pred_p_it[:,nn_ind_list]  = np.power(10, test_pred_p_it[:,nn_ind_list])

	#---------------------------------------------------;
	#  8. One-to_one normalized plots (train/val/test)  ;
	#---------------------------------------------------;
	str_x_label = 'Normalized ground truth'
	str_y_label = 'Normalized prediction'
	#
	for param_id in range(0,nswat_comps):
		x_train  = train_p[:,param_id]
		y_train  = train_pred_p[:,param_id]
		fl_name  = path_fl_sav + "P_train_"
		plot_gt_pred(x_train, y_train, param_id, fl_name, \
						str_x_label, str_y_label)
		#  
		x_val    = val_p[:,param_id]
		y_val    = val_pred_p[:,param_id]
		fl_name  = path_fl_sav + "P_val_"
		plot_gt_pred(x_val, y_val, param_id, fl_name, \
						str_x_label, str_y_label)
		#
		x_test   = test_p[:,param_id]
		y_test   = test_pred_p[:,param_id]
		fl_name  = path_fl_sav + "P_test_"
		plot_gt_pred(x_test, y_test, param_id, fl_name, \
						str_x_label, str_y_label)
	
	#----------------------------------------------------------;
	#  9. One-to_one inverse transform plots (train/val/test)  ;
	#----------------------------------------------------------;
	str_x_label = 'Ground truth'
	str_y_label = 'Prediction'
	#
	for param_id in range(0,nswat_comps):
		x_train  = train_p_it[:,param_id]
		y_train  = train_pred_p_it[:,param_id]
		fl_name  = path_fl_sav + "IT_train_"
		plot_gt_pred(x_train, y_train, param_id, fl_name, \
						str_x_label, str_y_label)
		#  
		x_val    = val_p_it[:,param_id]
		y_val    = val_pred_p_it[:,param_id]
		fl_name  = path_fl_sav + "IT_val_"
		plot_gt_pred(x_val, y_val, param_id, fl_name, \
						str_x_label, str_y_label)
		#
		x_test   = test_p_it[:,param_id]
		y_test   = test_pred_p_it[:,param_id]
		fl_name  = path_fl_sav + "IT_test_"
		plot_gt_pred(x_test, y_test, param_id, fl_name, \
						str_x_label, str_y_label)

	#-----------------------------;
	#  10. CNN-based calibration  ;
	#-----------------------------;
	xt_q_obs                     = qq_scalar.transform(x_q_obs) #scaled obs. q-data
	obs_pred_p                   = inv_model.predict(xt_q_obs) #CNN-predictions
	obs_pred_p_it                = pp_scalar.inverse_transform(obs_pred_p) #Inverse transform CNN-predictions
	#obs_pred_p_it[:,nn_ind_list] = np.power(10, obs_pred_p_it[:,nn_ind_list]) #Inverse log10 transform
	df_obs_p                     = pd.DataFrame(obs_pred_p_it, index = ['InvCNNModel-' + str(counter)], \
												columns = swat_p_list) #Create pd dataframe for obs SWAT params
	df_obs_p.to_csv(path_fl_sav + "SWAT_obs_params.csv")

	#--------------------------------------------------------------;
	#  11. Save model (TensorFlow SavedModel format. *.h5 format)  ;
	#--------------------------------------------------------------;
	inv_model.save(path_fl_sav + "Inv_CNN_Model") #TensorFlow SavedModel format
	inv_model.save(path_fl_sav + "Inv_CNN_Model.h5") #h5 format

	#-------------------------------------------------------------;
	#  12. Done training and saving the DL model and its outputs  ;
	#-------------------------------------------------------------;
	print('-------------------------------------------------------------')
	print('            Trained: ' + str(counter) + '_model/            ',)
	print('-------------------------------------------------------------')

#**********************************************************;
#  Set paths, load preprocessed data, and dump .csv files  ;
#**********************************************************;
#
if __name__ == '__main__':

	#=========================;
	#  Start processing time  ;
	#=========================;
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

	#=========================================;
	#  4. MPI send and receive realz numbers  ;
	#=========================================;
	if rank == 0:
		for i in range(size-1,-1,-1):
			realz_id = [0]*num_total  #Realization list
			#
			for j in range(num_total):
				print(j + num_total*i + 1, realz_id)
				realz_id[j] = j + num_total*i + 1
			print('rank and realz_id = ', rank, realz_id)
			#
			if i > 0: 
				comm.send(realz_id, dest = i)
	else:
		realz_id = comm.recv(source = 0)
		#print('rank, realz_id = ', rank, realz_id)

	#==========================================;
	#  5. Run DL model training for each realz ;
	#==========================================;
	for k in realz_id:
		start_at_this_hpfolder = 1 #1, 18751, 21751, 24001, 25501 (1, 2, 3, 4, and 5-CNN layers)
		get_trained_models(k, start_at_this_hpfolder)
		print('rank, k, realz_id = ', rank, k, realz_id, k+start_at_this_hpfolder-1)

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)