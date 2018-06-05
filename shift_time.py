# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:46:41 2018

@author: kkrao
"""

import os 
import glob
import numpy as np
import pandas as pd
import pytz
import datetime
from tzwhere import tzwhere


pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)

df = pd.read_pickle('D:\Krishna\Project\codes\wp_from_radar\data\sar_sm')
df['obs_date_local'] = datetime.datetime.now()
ll = pd.read_csv('D:\Krishna\Project\codes\wp_from_radar\data\scan\query_locations.csv',index_col = 0)

tzwhere = tzwhere.tzwhere()
for index, row in ll.iterrows():
    if index == 2062:
        continue
    tz_str = tzwhere.tzNameAt(row['Latitude'], row['Longitude']) # Seville coordinates
    if tz_str:
        tz = pytz.timezone(tz_str)
        df.loc[df.site == str(index),'obs_date_local'] = \
        pd.to_datetime(df.loc[df.site == str(index),'obs_date']+[tz.utcoffset(dt) for dt in df.loc[df.site == str(index),'obs_date']])
        print('[INFO] Time shift done for site %s'%index)
    else:
        print('[INFO] Time shifting failed for site %s'%index)
        
df.obs_date_local.apply(lambda x: x.hour).hist()
df.to_pickle('D:\Krishna\Project\codes\wp_from_radar\data\sar_sm')
