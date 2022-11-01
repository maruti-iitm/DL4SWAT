# Create directories and hyperparameter input files for CNN-based inverse models
# (Create directories on PINKLADY and NERSC)
# Total number of CNN models trained = 26250
#  Sanity check:
#	https://stackoverflow.com/questions/14798220/how-can-i-search-sub-folders-using-glob-glob-module
#
# module load texlive
# module load python/3.9-anaconda-2021.11
# module load cray-hdf5-parallel
# conda activate /global/common/software/m1800/maruti_python/conda/myenv1
# export HDF5_USE_FILE_LOCKING=FALSE
#
# python get_dir_hp_cnn_20pss_nersc.py
# AUTHOR: Maruti Kumar Mudunuru

import numpy as np
import glob
import os
import time
import subprocess
import itertools

if __name__ == '__main__':

	#=========================;
	#  Start processing time  ;
	#=========================;
	tic = time.perf_counter()

	#----------------------------;
	#  1. Paths for directories  ;
	#----------------------------;
	path     = "/global/project/projectdirs/m1800/maruti/2_DL_SWAT_v3/"
	dir_path = path + "1_InvCNNModel_ss/"
	print(dir_path)
	#
	if not os.path.exists(dir_path): #Create if they dont exist
		os.makedirs(dir_path)

	num_cnn_folders = 26250
	#
	for i in range(1,num_cnn_folders+1):
		dir_path    = path + "1_InvCNNModel_ss/" + str(i) + "_model/"
		print(dir_path)
		#
		if not os.path.exists(dir_path): #Create if they dont exist
			os.makedirs(dir_path)

	#----------------------------------------------------------;
	#  2a. Create hyperparameters (1-CNN-layer): 18750 models  ;
	#----------------------------------------------------------;
	num_models         = 18750 # 25*750
	num_layers         = 1
	filter_sizes_list  = [256, 128, 64, 32, 16] #5
	kernel_sizes_list  = [32, 16, 8, 4, 2] #5
	pool_sizes_list    = [2, 2, 2, 2, 2]
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [50, 100, 200, 300, 400, 500] #6
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 1
	hp_1cnn_list = [] #18750
	#
	for filter_sizes in filter_sizes_list:
		for kernel_sizes in kernel_sizes_list:
			for dropout_value in dropout_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_1cnn_list.append([counter, \
													num_layers, \
													filter_sizes, \
													kernel_sizes, \
													pool_sizes_list[0], \
													dropout_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, filter_sizes, \
		kernel_sizes, pool_sizes, dropout_value, \
		lr_values, epochs, batch_size = hp_1cnn_list[i]
		#
		path_fl_sav = path + "1_InvCNNModel_ss/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('filter_sizes  = ' + str(filter_sizes))
		f.write('\n')
		f.write('kernel_sizes  = ' + str(kernel_sizes))
		f.write('\n')
		f.write('pool_sizes    = ' + str(pool_sizes))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#---------------------------------------------------------;
	#  2b. Create hyperparameters (2-CNN-layer): 3000 models  ;
	#---------------------------------------------------------;
	num_models         = 3000 # 4*750
	num_layers         = 2
	filter_sizes_list  = [[256, 128], \
							[128, 64], \
							[64, 32], \
							[32, 16]] #4
	kernel_sizes_list  = [[32, 16], \
							[16, 8], \
							[8, 4], \
							[4, 2]] #4
	pool_sizes_list    = [2, 2, 2, 2, 2]
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [50, 100, 200, 300, 400, 500] #6
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 18751
	hp_2cnn_list = [] #3000
	#
	for (filter_sizes, kernel_sizes) in zip(filter_sizes_list, kernel_sizes_list):
			for dropout_value in dropout_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_2cnn_list.append([counter, \
													num_layers, \
													filter_sizes, \
													kernel_sizes, \
													pool_sizes_list[0:2], \
													dropout_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, filter_sizes, \
		kernel_sizes, pool_sizes, dropout_value, \
		lr_values, epochs, batch_size = hp_2cnn_list[i]
		#
		path_fl_sav = path + "1_InvCNNModel_ss/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('filter_sizes  = ' + str(filter_sizes[0]) + ', ' + str(filter_sizes[1]))
		f.write('\n')
		f.write('kernel_sizes  = ' + str(kernel_sizes[0]) + ', ' + str(kernel_sizes[1]))
		f.write('\n')
		f.write('pool_sizes    = ' + str(pool_sizes[0]) + ', ' + str(pool_sizes[1]))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#---------------------------------------------------------;
	#  2c. Create hyperparameters (3-CNN-layer): 2250 models  ;
	#---------------------------------------------------------;
	num_models         = 2250 # 3*750
	num_layers         = 3
	filter_sizes_list  = [[256, 128, 64], \
							[128, 64, 32], \
							[64, 32, 16]] #3
	kernel_sizes_list  = [[32, 16, 8], \
							[16, 8, 4], \
							[8, 4, 2]] #3
	pool_sizes_list    = [2, 2, 2, 2, 2]
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [50, 100, 200, 300, 400, 500] #6
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 21751
	hp_3cnn_list = [] #2250
	#
	for (filter_sizes, kernel_sizes) in zip(filter_sizes_list, kernel_sizes_list):
			for dropout_value in dropout_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_3cnn_list.append([counter, \
													num_layers, \
													filter_sizes, \
													kernel_sizes, \
													pool_sizes_list[0:3], \
													dropout_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, filter_sizes, \
		kernel_sizes, pool_sizes, dropout_value, \
		lr_values, epochs, batch_size = hp_3cnn_list[i]
		#
		path_fl_sav = path + "1_InvCNNModel_ss/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('filter_sizes  = ' + str(filter_sizes[0]) + ', ' + str(filter_sizes[1]) \
									+ ', ' + str(filter_sizes[2]))
		f.write('\n')
		f.write('kernel_sizes  = ' + str(kernel_sizes[0]) + ', ' + str(kernel_sizes[1]) \
									+ ', ' + str(kernel_sizes[2]))
		f.write('\n')
		f.write('pool_sizes    = ' + str(pool_sizes[0]) + ', ' + str(pool_sizes[1]) \
									+ ', ' + str(pool_sizes[2]))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#---------------------------------------------------------;
	#  2d. Create hyperparameters (4-CNN-layer): 1500 models  ;
	#---------------------------------------------------------;
	num_models         = 1500 # 2*750
	num_layers         = 4
	filter_sizes_list  = [[256, 128, 64, 32], \
							[128, 64, 32, 16]] #2
	kernel_sizes_list  = [[32, 16, 8, 4], \
							[16, 8, 4, 2]] #2
	pool_sizes_list    = [2, 2, 2, 2, 2]
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [50, 100, 200, 300, 400, 500] #6
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 24001
	hp_4cnn_list = [] #1500
	#
	for (filter_sizes, kernel_sizes) in zip(filter_sizes_list, kernel_sizes_list):
			for dropout_value in dropout_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_4cnn_list.append([counter, \
													num_layers, \
													filter_sizes, \
													kernel_sizes, \
													pool_sizes_list[0:4], \
													dropout_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, filter_sizes, \
		kernel_sizes, pool_sizes, dropout_value, \
		lr_values, epochs, batch_size = hp_4cnn_list[i]
		#
		path_fl_sav = path + "1_InvCNNModel_ss/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('filter_sizes  = ' + str(filter_sizes[0]) + ', ' + str(filter_sizes[1]) \
									+ ', ' + str(filter_sizes[2]) + ', ' + str(filter_sizes[3]))
		f.write('\n')
		f.write('kernel_sizes  = ' + str(kernel_sizes[0]) + ', ' + str(kernel_sizes[1]) \
									+ ', ' + str(kernel_sizes[2]) + ', ' + str(kernel_sizes[3]))
		f.write('\n')
		f.write('pool_sizes    = ' + str(pool_sizes[0]) + ', ' + str(pool_sizes[1]) \
									+ ', ' + str(pool_sizes[2]) + ', ' + str(pool_sizes[3]))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#--------------------------------------------------------;
	#  2e. Create hyperparameters (5-CNN-layer): 750 models  ;
	#--------------------------------------------------------;
	num_models         = 750 #750
	num_layers         = 5
	filter_sizes_list  = [[256, 128, 64, 32, 16]] #1
	kernel_sizes_list  = [[32, 16, 8, 4, 2]] #1
	pool_sizes_list    = [2, 2, 2, 2, 2]
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [50, 100, 200, 300, 400, 500] #6
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 25501
	hp_5cnn_list = [] #750
	#
	for (filter_sizes, kernel_sizes) in zip(filter_sizes_list, kernel_sizes_list):
			for dropout_value in dropout_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_5cnn_list.append([counter, \
													num_layers, \
													filter_sizes, \
													kernel_sizes, \
													pool_sizes_list, \
													dropout_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, filter_sizes, \
		kernel_sizes, pool_sizes, dropout_value, \
		lr_values, epochs, batch_size = hp_5cnn_list[i]
		#
		path_fl_sav = path + "1_InvCNNModel_ss/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('filter_sizes  = ' + str(filter_sizes[0]) + ', ' + str(filter_sizes[1]) \
									+ ', ' + str(filter_sizes[2]) + ', ' + str(filter_sizes[3]) \
									+ ', ' + str(filter_sizes[4]))
		f.write('\n')
		f.write('kernel_sizes  = ' + str(kernel_sizes[0]) + ', ' + str(kernel_sizes[1]) \
									+ ', ' + str(kernel_sizes[2]) + ', ' + str(kernel_sizes[3]) \
									+ ', ' + str(kernel_sizes[4]))
		f.write('\n')
		f.write('pool_sizes    = ' + str(pool_sizes[0]) + ', ' + str(pool_sizes[1]) \
									+ ', ' + str(pool_sizes[2]) + ', ' + str(pool_sizes[3]) \
									+ ', ' + str(pool_sizes[4]))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#--------------------------;
	#  3. Test some scenarios  ;
	#--------------------------;
	counter_list  = [1, 18751, 21751, 24001, 25501]
	#
	for counter in counter_list:
		path_fl_sav   = path + "1_InvCNNModel_ss/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		fl_id         = open(path_fl_sav) #Read the hp file line by line
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

	#-----------------------------------------------;
	#  4. Sanity check on hp .txt files using glob  ;
	#-----------------------------------------------;
	hp_glob_txt_list = glob.glob(path + '1_InvCNNModel_ss/**/*.txt', recursive = True) #26250
	print(len(hp_glob_txt_list))

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)