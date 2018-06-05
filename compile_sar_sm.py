# -*- coding: utf-8 -*-
"""
Created on Sun May 27 03:36:44 2018

@author: kkrao
"""

import os 
import glob
import pandas as pd
import numpy as np
from datetime import timedelta



def find_nearest(known_array = None, test_array = None):
    known_array = np.array(known_array)
    test_array  = np.array(test_array)
    
    differences = (test_array.reshape(1,-1) - known_array.reshape(-1,1))
    indices = np.abs(differences).argmin(axis=0)
    residual = np.diagonal(differences[indices,])
    residual = residual.astype('timedelta64[D]')/ np.timedelta64(1, 'D')
    residual = residual.astype(int)
    
    return indices, residual

def compile_sar_sm():
    ###########################################################################
    os.chdir("D:\Krishna\Project\codes\wp_from_radar\data\scan\sm")
    
    files = [i for i in glob.glob('*.{}'.format("csv"))]
    Df = pd.DataFrame()
    
    for file in files:
        df = pd.read_csv(file, skiprows = 280, index_col=None, dtype = {'Station Id':str})
        Df = pd.concat([Df,df], axis =0, sort = False)
    
    Df.columns = Df.columns.map(lambda x: x.lstrip("Soil Moisture Percent").\
                   rstrip('(pct) Mean of Hourly Values') if 'Soil Moisture Percent' \
                   in x else x)
    Df.columns = Df.columns.map(lambda x: x.lstrip("-") if '-' in x else x)
    Df.columns = Df.columns.map(lambda x: x+'n' if x[-1] == 'i' else x)
    Df.rename(columns = {'Station Id': 'site'}, inplace = True)
    Df.drop(['Station Name','24in', '26in','33in','35in','43in', '51in'],\
            inplace = True, axis = 1)
#    print(Df.head())
    meas = Df.copy()
#    meas.site = meas.site.astype('str')
    meas.rename(columns = {"Date":"meas_date"}, inplace = True)
    meas.meas_date = pd.to_datetime(meas.meas_date)
    
    ###########################################################################
    os.chdir("D:\Krishna\Project\codes\wp_from_radar\data\sentinel1\VV-VH")
    files = [i for i in glob.glob('*.{}'.format("csv"))]
    Df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=None, dtype = {'VV':np.float16,'VH':np.float16})
        df = df.loc[:,['VV','VH', 'date']]
        df['site'] = file.rstrip('_S1_GRD_gee_subset.csv')
        Df = pd.concat([Df,df], axis =0, sort = False)
        
    os.chdir("D:\Krishna\Project\codes\wp_from_radar\data\sentinel1\VV")
    files = [i for i in glob.glob('*.{}'.format("csv"))]
    for file in files:
        df = pd.read_csv(file, index_col=None, dtype = {'VV':np.float16})
        df = df.loc[:,['VV', 'date']]
        df['site'] = file.rstrip('_S1_GRD_gee_subset.csv')
        df['VH'] = np.nan
        Df = pd.concat([Df,df], axis =0, sort = False)

    obs = Df.copy()
    obs.rename(columns = {"date":"obs_date"}, inplace = True)
    obs.obs_date = pd.to_datetime(obs.obs_date)
    obs.reset_index(inplace = True, drop = True)

    ###########################################################################
    meas.reset_index(inplace = True, drop = True)
    meas['VV'] = np.nan; meas['VH'] = np.nan; 
    meas['obs_date'] = np.nan; meas['residual'] = np.nan
    for site in meas.site.unique():
        print('[INFO] Finding match for site %s'%site)
        if site in obs.site.unique():
            obs_sub = obs.loc[obs['site'] == site,:]
            indices, residual = find_nearest(\
                                 obs_sub.obs_date,\
                                 meas.loc[meas['site'] == site,'meas_date'] )
            
            meas.loc[meas['site'] == site,['VV','VH','obs_date']] = \
                obs_sub.loc[obs_sub.index[indices],['VV','VH','obs_date']].values
            meas.loc[meas['site'] == site,'residual'] = residual
    
    meas.to_pickle("D:\Krishna\Project\codes\wp_from_radar\data\sar_sm")
    return meas

def main():
    compile_sar_sm()

if __name__ == '__main__':
    main()
            
