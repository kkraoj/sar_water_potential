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
filter = (df.residual.abs() <=20) \
            &(df.obs_date_local.apply(lambda x: x.hour)<=8)
d = df.loc[filter,:]