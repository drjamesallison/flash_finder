import os
import numpy as np
from fitting import *

# Initialise required directories
def initialize_directories(options):
    os.makedirs('%s'%options.data_path, exist_ok=True)
    os.makedirs('%s'%options.out_path, exist_ok=True)

# Create list of file info objects for files of particular extensions
def listDirectory(directory, fileExtList):
    fileList = [os.path.normcase(f)
                for f in os.listdir(directory)]
    fileList = [os.path.join(directory, f)
                for f in fileList
                if os.path.splitext(f)[1] in fileExtList]
    return fileList

# Initialize MultiNest arguments
def initialize_mnest(options,source,model):
    mnest_args = {'LogLikelihood':LogLGen(options,source,model),
                  'Prior':PriorGen(options,model),
                  'n_dims':0,
                  'n_params':0,
                  'n_clustering_params':0,
                  'wrapped_params':None,
                  'importance_nested_sampling':(not options.mmodal),
                  'multimodal':options.mmodal,
                  'const_efficiency_mode':options.ceff,
                  'n_live_points':int(options.nlive),
                  'evidence_tolerance':float(options.tol),
                  'sampling_efficiency':float(options.efr),
                  'n_iter_before_update':1000,
                  'null_log_evidence':model.output.null.logZ,
                  'max_modes':100,
                  'mode_tolerance':-1.e90,
                  'outputfiles_basename':'',
                  'seed':-1,
                  'verbose':options.verbose,
                  'resume':True,
                  'context':0,
                  'write_output':True,
                  'log_zero':-1.e100,
                  'max_iter':0,
                  'init_MPI':options.init_MPI,
                  'dump_callback':None}
    return mnest_args

# Estimate the standard deviation using MADFM statistic
def std_madfm(data):
    absdev = np.abs(np.subtract(data, np.median(data)))
    sigma = 1.4826042*np.median(absdev)
    return sigma 
