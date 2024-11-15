#!/home/carlau/.conda/envs/test_env_2_leach/bin/python3

#SBATCH --time=10:00:00 --cpus-per-task=1 --hint=nomultithread --mem=32G --partition=normal 

# add the local directory to the path
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import scipy as sp
from tqdm import tqdm
import glob
import xarray as xr

# from fair import *
# from fair.scripts.stats import *

# change directory to the notebook direcotory
import os
os.chdir('/home/carlau/ACAI_testing/create_temperature_ensamble/leach-et-al-2021/notebooks/perturbed-parameter-ensemble')

import sys
sys.path.append('/home/carlau/ACAI_testing/create_temperature_ensamble/leach-et-al-2021/fair/')
sys.path.append('/home/carlau/ACAI_testing/create_temperature_ensamble/leach-et-al-2021/fair/scripts')

from fair_runner import *
from stats import *


# ################### First set of calculations
# ## get temperature timeseries
# ant_temps = xr.open_dataarray('../../aux/output-data/global-warming-index/ant_temperature.nc',chunks={'forcing_mem':100})
# T_2010_2019 = (ant_temps.sel(time=slice('2010','2019')).mean('time')-ant_temps.sel(time=slice('1850','1900')).mean('time')).astype(np.single)
# dT_2010_2019 = ant_temps.sel(time=slice('2010','2019')).assign_coords(time=np.arange(10)).polyfit(dim='time',deg=1).polyfit_coefficients.sel(degree=1).astype(np.single)


# ## load scaling coefficients and compute level/rate (very memory intensive)

# # 500 million member subsamples to start = 2GB on disk per quantity
# sub_size = int(5e8)
# subsamples={}
# ## HadCRUT5 subsample
# subsamples['HadCRUT5_obs_mem'] = np.random.choice(5000*102*18*200,sub_size)
# ## HadCRUT4 subsample
# subsamples['HadCRUT4_obs_mem'] = np.random.choice(5000*102*18*100,sub_size)
# ## CW subsample
# subsamples['CW_obs_mem'] = np.random.choice(5000*102*18*99,sub_size)


# for select_dataset in ['HadCRUT5','HadCRUT4','NOAA','GISTEMP','CW','BERKELEY']:
#     print(select_dataset)
#     ant_coefs = (xr.open_dataarray('../../aux/output-data/global-warming-index/ant_coefs_forc_'+select_dataset+'.nc',chunks={'forcing_mem':100}).astype(np.single)+\
#                  xr.open_dataarray('../../aux/output-data/global-warming-index/ant_coefs_IV_'+select_dataset+'.nc',chunks={'forcing_mem':100}).astype(np.single))
    
#     obsv_unc_source = ant_coefs.dims[0]
    
#     np.save('../../aux/output-data/global-warming-index/results/T_2010-2019_'+select_dataset+'.npy',(T_2010_2019*ant_coefs).values.flatten()[subsamples[obsv_unc_source]])
#     np.save('../../aux/output-data/global-warming-index/results/dT_2010-2019_'+select_dataset+'.npy',(dT_2010_2019*ant_coefs).values.flatten()[subsamples[obsv_unc_source]])


################### Second set of calculations
FULL_metrics = pd.read_hdf('../../aux/parameter-sets/perturbed-parameters/FULL_ANT.h5')

FULL_level = FULL_metrics.T_2010_2019 - FULL_metrics.T_1850_1900
FULL_rate = FULL_metrics.dT_2010_2019

# ALT_metrics = pd.read_hdf('../../aux/parameter-sets/perturbed-parameters/ALT_ANT.h5')

# ALT_level = ALT_metrics.T_2010_2019 - ALT_metrics.T_1850_1900
# ALT_rate = ALT_metrics.dT_2010_2019

## choose resolution of bins here (delibarately large bins)
level_bins = np.arange(-0.2,1.8,0.01)
rate_bins = np.arange(-0.05,0.1,0.001)

for select_dataset in ['HadCRUT5','HadCRUT4','NOAA','GISTEMP','CW','BERKELEY']:
    print('calculating selection probabilities for '+select_dataset, flush=True)
    
    ## import the pre-computed AWI level & rate
    warming_level = np.load('../../aux/output-data/global-warming-index/results/T_2010-2019_'+select_dataset+'.npy')
    warming_rate = np.load('../../aux/output-data/global-warming-index/results/dT_2010-2019_'+select_dataset+'.npy')
    
    ## bin the data in 2d
    AWI_binned = sp.stats.binned_statistic_2d(warming_level,warming_rate,None,'count',bins=[level_bins,rate_bins])
    AWI_likelihood = AWI_binned.statistic / AWI_binned.statistic.max()
    
    ## create a dataseries to store the member probabilities
    FULL_probabilities = pd.Series(index=FULL_rate.index,dtype=float)
    # ALT_probabilities = pd.Series(index=ALT_rate.index,dtype=float)
    
    ## set values outside the AWI max/min values to have 0 probability
    FULL_probabilities.loc[(FULL_level>warming_level.max())|(FULL_level<-warming_level.min())|(FULL_rate>warming_rate.max())|(FULL_rate<warming_rate.min())] = 0
    # ALT_probabilities.loc[(ALT_level>warming_level.max())|(ALT_level<-warming_level.min())|(ALT_rate>warming_rate.max())|(ALT_rate<warming_rate.min())] = 0

    FULL_binned = sp.stats.binned_statistic_2d(FULL_level.loc[FULL_probabilities.isna()],FULL_rate.loc[FULL_probabilities.isna()],None,'count',bins=[level_bins,rate_bins],expand_binnumbers=True)
    # ALT_binned = sp.stats.binned_statistic_2d(ALT_level.loc[ALT_probabilities.isna()],ALT_rate.loc[ALT_probabilities.isna()],None,'count',bins=[level_bins,rate_bins],expand_binnumbers=True)
    
    ## have to reduce binnumbers by 2 as scipy adds one boundary bin, and bins start from 1
    FULL_probabilities.loc[FULL_probabilities.isna()] = AWI_likelihood[FULL_binned.binnumber[0]-2,FULL_binned.binnumber[1]-2]
    # ALT_probabilities.loc[ALT_probabilities.isna()] = AWI_likelihood[ALT_binned.binnumber[0]-2,ALT_binned.binnumber[1]-2]
    
    ## save the probabilities
    FULL_probabilities.to_hdf(r'../../aux/parameter-sets/perturbed-parameters/FULL_selection_probability-'+select_dataset+'.h5', key='stage', mode='w')
    # ALT_probabilities.to_hdf(r'../../aux/parameter-sets/perturbed-parameters/ALT_selection_probability-'+select_dataset+'.h5', key='stage', mode='w')



print("Completed successfully", flush=True)





