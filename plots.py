# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:27:49 2018

@author: kkrao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from scipy import stats

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2, stats.pearsonr(x, y)[1]

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)
sns.set(font_scale = 1, style = 'ticks')


df = pd.read_pickle('D:\Krishna\Project\codes\wp_from_radar\data\sar_sm')
#df.to_pickle('D:\Krishna\Project\codes\wp_from_radar\data\sar_sm')
#df.obs_date_local = pd.to_datetime(df.obs_date_local, format = '%Y-%m-%d %H:%M')
###############################################################################
sites_2in = ['2205', '2040', '2002', '808']
sites_8in = ['2002', '2197', '2172', '2160'] #>5pm
sites_20in = ['2104','2093', '2169', '2160']
###############################################################################
sand = 46.
clay = 16.
A = 100*np.exp(-4.396 -0.0715*clay-4.880*10**(-4)*sand**2-4.285*10**(-5)*sand**2*clay)
B = -3.140 - 0.00222*clay**2 - 3.484*10**(-5)*sand**2*clay 
###############################################################################


#for col in ['2in'  , '4in'  , '8in'  ,'20in'  ,'40in']:
#    df.loc[df[col]<10,col]= np.nan
#    df.loc[df[col]>70,col]= np.nan
for col in ['2in'  , '4in'  , '8in'  ,'20in'  ,'40in']:
    df['p'+col]=-A*(df[col]/100)**B
df = df.round(2)
#for col in ['2in'  , '4in'  , '8in'  ,'20in'  ,'40in']:
#    df['p'+col]=-df['p'+col]
###############################################################################
df = df.loc[df.residual.abs() <=10,:]
fig, ax = plt.subplots()
filter = (df.site.isin(sites_20in+sites_2in+sites_8in))
d = df.loc[filter,:]


d.hist(column = 'residual', bins = 20, facecolor='g', alpha=0.75, ax = ax)
ax.set_xlabel('$\Delta$ delay (days)')
ax.set_ylabel('Frequency (-)')
ax.set_title('')
ax.grid('off')
ax.axvline(x=-6, ls = '--', c='r')
ax.axvline(x=6, ls = '--', c='r')
###############################################################################
#filter = (df.site == '2205')
#
#d = df.loc[filter,:]
#
#fig, ax = plt.subplots()
#d.plot(x = 'meas_date', y = ['2in','4in','8in','20in','40in'], style='.', ax = ax)
#ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
###############################################################################

###############################################################################
filter =\
    (df.site.isin(sites_8in))\
    &(df.residual.abs() <=10) \
    &(df.obs_date_local.dt.hour>17)\
    &(df.obs_date_local.dt.year>=2016)\
            &(df.obs_date_local.dt.month>=6)\
            &(df.obs_date_local.dt.month<9)
d = df.loc[filter,:]

fig, ax = plt.subplots()
#d.plot.scatter(x = 'p2in', y = 'VV', \
#              ylim = [-20,-5], xlim = [-10**4, -10**(-1)], ax = ax)
ax.set_xscale('symlog')
#ax.set_yscale('symlog')
sns.regplot(x="p20in", y="VV", data=d, ax = ax, order = 1, color = 'darkgreen')
ax.set_ylabel('VV (dB)')
ax.set_xlabel('$\psi$ (KPa)')
ax.set_title('Trees')
ax.set_ylim([-20,-5])
ax.set_xlim([-10**2.5, -10**0])
r2 = d['p20in'].corr(d['VV'])**2
ax.annotate('$R^2$ = %0.2f'%r2, xy=(0.65, 0.15), xycoords='axes fraction')

y=d['VV']; x=d['p8in']
non_nan_ind=np.where(~np.isnan(x))[0]
x=x.take(non_nan_ind);y=y.take(non_nan_ind)
non_nan_ind=np.where(~np.isnan(y))[0]
x=x.take(non_nan_ind);y=y.take(non_nan_ind)
stats.pearsonr(x, y)


###############################################################################
#for site in [sites_20in[2]]:
#    filter =\
#            (df.site == site)\
#            &(df.residual.abs() <=6) \
#            &(df.obs_date_local.dt.hour>17)\
#            &(df.obs_date_local.dt.year>=2016)\
##            &(df.obs_date_local.dt.month>=6)\
##            &(df.obs_date_local.dt.month<9)
#
#    d = df.loc[filter,:]
#    if d.shape[0]<30:
#        continue
#    fig, ax = plt.subplots()
#    ax.plot(d.obs_date_local, d['p2in'].rolling(4).mean(), 'b.')
#    ax.set_yscale('symlog')
#    ax.set_ylim([-10**4, -10**(-1)])
#    ax.set_ylabel('$\psi (KPa) $', color='b')
#    ax.tick_params('y', colors='b')
##    fmt = '-{x:, .0f}'
##    tick = mtick.StrMethodFormatter(fmt)
##    ax.yaxis.set_major_formatter(tick) 
##    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,-1))
#    ax2 = ax.twinx()
#    ax2.plot(d.obs_date_local, d.VV.rolling(3).mean(), 'm.')
#    ax2.set_ylim([-30,0])
#    ax2.set_ylabel('VV (dB)', color = 'm')
#    ax2.tick_params('y', colors='m')
#    fig.autofmt_xdate()
##    d.plot(x = 'obs_date_local', y = ['8in','VV'], style=['-','--'], ax = ax)
##    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
##    ax.set_title('Site %s'%site)
##    ax.set_xlim([-25,0])
#    plt.show()
    
###############################################################################
#x = np.linspace(0.1,0.5)

#y2=A*x**B
    
    
#sand = 26.
#clay = 45.
#A = 0.1*np.exp(-4.396 -0.0715*clay-4.880*10**(-4)*sand**2-4.285*10**(-5)*sand**2*clay)
#B = -3.140 - 0.00222*clay**2 - 3.484*10**(-5)*sand**2*clay 
#y1=A*x**B
#
#sand = 6.
#clay = 60.
#A = 0.1*np.exp(-4.396 -0.0715*clay-4.880*10**(-4)*sand**2-4.285*10**(-5)*sand**2*clay)
#B = -3.140 - 0.00222*clay**2 - 3.484*10**(-5)*sand**2*clay 
#y3=A*x**B
#
#y = [y3, y1, y2]
#labels = ['clayey', 'mixed', 'sandy']
#
#fig,ax = plt.subplots()
#for y_arr, label in zip(y, labels):
#    ax.semilogy(x, y_arr, label=label)
#ax.set_xlabel(r'$\theta$ ($m^3/m^3$)')
#ax.set_ylabel('$\psi$ (KPa)')
#plt.legend()
#plt.show()
# deep_sites = ['2169',] # 20in