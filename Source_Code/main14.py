# Smart Hedging Main Routine

import os
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pylab import rcParams
import statsmodels.api as sm
import itertools
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric

from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
# Using TensorFlow backend.
from keras.layers import Dense
from numpy import array
from numpy import mean
from numpy import std

wdir = os.getcwd()
datadir_yc = os.path.join(wdir,"data")
datadir_fas = os.path.join(wdir,"data\TPC")

file_chflibor = "/chflibor_monthly.xlsx"
full_path_chflibor = datadir_yc+file_chflibor

chflibor_temp = pd.read_excel(full_path_chflibor
                        ,sheet_name = 'CHFLIBOR'
                        ,usecols=['VAL_DATE','DAYS_FWD','ZERO_RATE'])

#chflibor = pd.DataFrame(chflibor_temp[(chflibor_temp['VAL_DATE'].dt.date >= pd.to_datetime("2000-12-31"))
#                                    & (chflibor_temp['VAL_DATE'].dt.month == 12)])

chflibor = pd.DataFrame(chflibor_temp[(chflibor_temp['VAL_DATE'].dt.date >= pd.to_datetime("2017-12-31"))])

chflibor['VAL_DATE'] = chflibor['VAL_DATE'].dt.date

dates_yc = chflibor['VAL_DATE'].unique()
dfwd_max = chflibor['DAYS_FWD'].max()
chflibor_all = pd.DataFrame(index=range(1,dfwd_max+1), columns=dates_yc)
chflibor_all = chflibor_all.add_prefix('ZERO_RATE_')

ir = 0

for ir, row in chflibor.iterrows():
    current_date_yc = row[0]
    dfwd = row[1]
    rate = row[2]
    
    target_column_yc = "ZERO_RATE_" + str(current_date_yc)
    
    chflibor_all.at[dfwd,target_column_yc] = rate

chflibor_all = chflibor_all.apply(pd.to_numeric,errors='coerce')
chflibor_all_ipl = chflibor_all.interpolate(method='linear',limit_direction='forward',axis=0)
chflibor_all_ipl = chflibor_all_ipl.fillna(method='backfill',axis=0)

plt.figure(num=1)
plt.scatter(chflibor_all.index,chflibor_all['ZERO_RATE_2017-12-31'],label="ZERO_RATE_2017-12-31",s=2)
plt.title("CHF Libor")
plt.xlabel("Days Forward")
plt.ylabel("Zero Rate")
plt.legend()
plt.show()

plt.figure(num=2)
plt.scatter(chflibor_all_ipl.index,chflibor_all_ipl['ZERO_RATE_2017-12-31'],label="ZERO_RATE_2017-12-31",s=2)
plt.title("CHF Libor Interpolated")
plt.xlabel("Days Forward")
plt.ylabel("Zero Rate")
plt.legend()
plt.show()

chflibor_all_ipl_deltas                 = chflibor_all_ipl.diff(axis=1)
chflibor_all_ipl_pct_deltas             = chflibor_all_ipl.pct_change(axis=1)

chflibor_all_ipl_deltas_sum_dfwd        = chflibor_all_ipl_deltas.sum(axis=1)
chflibor_all_ipl_pct_deltas_sum_dfwd    = chflibor_all_ipl_pct_deltas.sum(axis=1)

chflibor_all_ipl_deltas_avg_monthly     = chflibor_all_ipl_deltas.mean(axis=0)
chflibor_all_ipl_pct_deltas_avg_monthly = chflibor_all_ipl_pct_deltas.mean(axis=0)

chflibor_all_ipl_deltas_avg_monthly     = pd.DataFrame(chflibor_all_ipl_deltas_avg_monthly).transpose()
chflibor_all_ipl_pct_deltas_avg_monthly = pd.DataFrame(chflibor_all_ipl_pct_deltas_avg_monthly).transpose()

chflibor_all_ipl_deltas_sum_dfwd        = pd.DataFrame(chflibor_all_ipl_deltas_sum_dfwd)
chflibor_all_ipl_pct_deltas_sum_dfwd    = pd.DataFrame(chflibor_all_ipl_pct_deltas_sum_dfwd)

fas133_data = "/fas133_data_*.xlsx"
full_path_fas133_data = datadir_fas+fas133_data

all_fas_files = glob.glob(full_path_fas133_data)
all_fas_files.sort()
#all_fas_files = [datadir_fas+"/fas133_data_20200228.xlsx",datadir_fas+"/fas133_data_20200331.xlsx",datadir_fas+"/fas133_data_20200430.xlsx"]

ind = 0
prev_date_fas = '0001-01-01'

for filename in all_fas_files:
    ind += 1
    fas133_data_temp = pd.read_excel(filename
                                    ,sheet_name = 'Sheet1'
                                    ,usecols=['COB Date'
                                             ,'Deal No','CCY','L/S'
                                             ,'Start Date','End Date','FREQUENCY'
                                             ,'RATE_TYPE','RATE_REF','INTERNAL_RATE','GCOC_RATE'
                                             ,'Notional'
                                             ,'CLEAN_NPV','EVENT_02'
                                             ])
    
    all_swaps_fixed_legs = fas133_data_temp.loc[fas133_data_temp['RATE_TYPE'] == "FIXED"]
    all_swaps_float_legs = fas133_data_temp.loc[fas133_data_temp['RATE_TYPE'] == "FLOAT"]
    
    all_swaps_flattened = pd.merge(all_swaps_fixed_legs,all_swaps_float_legs
                                  ,how='outer',on='Deal No'
                                  ,left_index=False,right_index=False,sort=False
                                  ,suffixes=('_FIXED', '_FLOAT'),copy=True,indicator=False
                                  ,validate="1:1")
    
    chf_swaps_flattened = all_swaps_flattened.loc[(all_swaps_flattened['CCY_FIXED'] == "CHF")
                                               & ((all_swaps_flattened['RATE_REF_FLOAT'] == "3MLIBOR")  |
                                                  (all_swaps_flattened['RATE_REF_FLOAT'] == "6MLIBOR"))
                                                 ]

    chf_swaps_flattened_tf = chf_swaps_flattened[['Deal No'
                                                 ,'Start Date_FIXED'
                                                 ,'End Date_FIXED'
                                                 ,'Notional_FIXED'
                                                 ,'L/S_FIXED'
                                                 ,'L/S_FLOAT'
                                                 ,'FREQUENCY_FIXED'
                                                 ,'FREQUENCY_FLOAT'
                                                 ,'RATE_REF_FLOAT'
                                                 ,'INTERNAL_RATE_FIXED'
                                                 ,'INTERNAL_RATE_FLOAT'
                                                 ,'GCOC_RATE_FIXED'
                                                 ,'COB Date_FIXED'
                                                 ,'COB Date_FLOAT'
                                                ]].copy()
    
    chf_swaps_flattened_tf = chf_swaps_flattened_tf.rename(columns={'Deal No':'DEAL_NO'
                                                                   ,'Start Date_FIXED':'START_DATE'
                                                                   ,'End Date_FIXED':'END_DATE'
                                                                   ,'Notional_FIXED':'NOTIONAL'
                                                                   ,'RATE_REF_FLOAT':'RATE_REF'
                                                                   ,'GCOC_RATE_FIXED':'GCOC_RATE'
                                                                   ,'COB Date_FIXED':'COB_DATE_FIRST'
                                                                   ,'COB Date_FLOAT':'COB_DATE_LAST'
                                                                   })
    
    chf_swaps_flattened_tf['START_DATE']     = chf_swaps_flattened_tf['START_DATE'].dt.date
    chf_swaps_flattened_tf['END_DATE']       = chf_swaps_flattened_tf['END_DATE'].dt.date
    chf_swaps_flattened_tf['COB_DATE_FIRST'] = chf_swaps_flattened_tf['COB_DATE_FIRST'].dt.date
    chf_swaps_flattened_tf['COB_DATE_LAST']  = chf_swaps_flattened_tf['COB_DATE_LAST'].dt.date
    
    chf_swaps_flattened_tf = chf_swaps_flattened_tf.set_index('DEAL_NO')
     
    chf_swaps_flattened_tf['COB_DATE_LAST'] = '9999-12-31'

    chf_swaps_flattened_ts_pv_fix = chf_swaps_flattened[['Deal No','CLEAN_NPV_FIXED']].copy()
    chf_swaps_flattened_ts_pv_flt = chf_swaps_flattened[['Deal No','CLEAN_NPV_FLOAT']].copy()
    chf_swaps_flattened_ts_ev_fix = chf_swaps_flattened[['Deal No','EVENT_02_FIXED']].copy()
    chf_swaps_flattened_ts_ev_flt = chf_swaps_flattened[['Deal No','EVENT_02_FLOAT']].copy()

    current_date_fas = chf_swaps_flattened_tf['COB_DATE_FIRST'].iloc[0]
    target_column_pv_fix = "CLEAN_NPV_FIXED_" + str(current_date_fas)
    target_column_pv_flt = "CLEAN_NPV_FLOAT_" + str(current_date_fas)
    target_column_ev_fix = "EVENT_02_FIXED_"  + str(current_date_fas)
    target_column_ev_flt = "EVENT_02_FLOAT_"  + str(current_date_fas)

    chf_swaps_flattened_ts_pv_fix = chf_swaps_flattened_ts_pv_fix.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'CLEAN_NPV_FIXED': target_column_pv_fix})
    chf_swaps_flattened_ts_pv_flt = chf_swaps_flattened_ts_pv_flt.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'CLEAN_NPV_FLOAT':target_column_pv_flt})
    chf_swaps_flattened_ts_ev_fix = chf_swaps_flattened_ts_ev_fix.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'EVENT_02_FIXED':target_column_ev_fix})
    chf_swaps_flattened_ts_ev_flt = chf_swaps_flattened_ts_ev_flt.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'EVENT_02_FLOAT': target_column_ev_flt})
        
    if ind == 1:
        chf_swaps_flattened_tf_all = chf_swaps_flattened_tf
        chf_swaps_flattened_ts_pv_fix_all = chf_swaps_flattened_ts_pv_fix
        chf_swaps_flattened_ts_pv_flt_all = chf_swaps_flattened_ts_pv_flt
        chf_swaps_flattened_ts_ev_fix_all = chf_swaps_flattened_ts_ev_fix
        chf_swaps_flattened_ts_ev_flt_all = chf_swaps_flattened_ts_ev_flt
    else:
        chf_swaps_flattened_tf_to_add = chf_swaps_flattened_tf.loc[~chf_swaps_flattened_tf.index.isin(chf_swaps_flattened_tf_all.index)]
        chf_swaps_flattened_tf_missing = chf_swaps_flattened_tf_all.loc[(~chf_swaps_flattened_tf_all.index.isin(chf_swaps_flattened_tf.index))
                                                                      & (chf_swaps_flattened_tf_all['COB_DATE_LAST'] == '9999-12-31')]
        chf_swaps_flattened_tf_all = chf_swaps_flattened_tf_all.append(chf_swaps_flattened_tf_to_add
                                                                      ,ignore_index=False,verify_integrity=True,sort=False)
        chf_swaps_flattened_tf_all.loc[chf_swaps_flattened_tf_all.index.isin(chf_swaps_flattened_tf_missing.index),'COB_DATE_LAST'] = prev_date_fas
        
        chf_swaps_flattened_ts_pv_fix_all = pd.merge(chf_swaps_flattened_ts_pv_fix_all,chf_swaps_flattened_ts_pv_fix
                                                    ,how='outer',on='DEAL_NO'
                                                    ,left_index=False,right_index=False,sort=False
                                                    ,suffixes=(False,False),copy=True,indicator=False
                                                    ,validate="1:1")

        chf_swaps_flattened_ts_pv_flt_all = pd.merge(chf_swaps_flattened_ts_pv_flt_all,chf_swaps_flattened_ts_pv_flt
                                                    ,how='outer',on='DEAL_NO'
                                                    ,left_index=False,right_index=False,sort=False
                                                    ,suffixes=(False,False),copy=True,indicator=False
                                                    ,validate="1:1")
        
        chf_swaps_flattened_ts_ev_fix_all = pd.merge(chf_swaps_flattened_ts_ev_fix_all,chf_swaps_flattened_ts_ev_fix
                                                    ,how='outer',on='DEAL_NO'
                                                    ,left_index=False,right_index=False,sort=False
                                                    ,suffixes=(False,False),copy=True,indicator=False
                                                    ,validate="1:1")

        chf_swaps_flattened_ts_ev_flt_all = pd.merge(chf_swaps_flattened_ts_ev_flt_all,chf_swaps_flattened_ts_ev_flt
                                                    ,how='outer',on='DEAL_NO'
                                                    ,left_index=False,right_index=False,sort=False
                                                    ,suffixes=(False,False),copy=True,indicator=False
                                                    ,validate="1:1")
        
        chf_swaps_flattened_ts_pv_fix_all = chf_swaps_flattened_ts_pv_fix_all.set_index('DEAL_NO')
        chf_swaps_flattened_ts_pv_flt_all = chf_swaps_flattened_ts_pv_flt_all.set_index('DEAL_NO')
        chf_swaps_flattened_ts_ev_fix_all = chf_swaps_flattened_ts_ev_fix_all.set_index('DEAL_NO')
        chf_swaps_flattened_ts_ev_flt_all = chf_swaps_flattened_ts_ev_flt_all.set_index('DEAL_NO')
        
        chf_swaps_flattened_ts_pv_fix_all_deltas = chf_swaps_flattened_ts_pv_fix_all.diff(axis=1)
        chf_swaps_flattened_ts_pv_flt_all_deltas = chf_swaps_flattened_ts_pv_flt_all.diff(axis=1)
        chf_swaps_flattened_ts_ev_fix_all_deltas = chf_swaps_flattened_ts_ev_fix_all.diff(axis=1)
        chf_swaps_flattened_ts_ev_flt_all_deltas = chf_swaps_flattened_ts_ev_flt_all.diff(axis=1)
        
        chf_swaps_flattened_ts_pv_fix_all_pct_deltas = chf_swaps_flattened_ts_pv_fix_all.pct_change(axis=1)
        chf_swaps_flattened_ts_pv_flt_all_pct_deltas = chf_swaps_flattened_ts_pv_flt_all.pct_change(axis=1)
        chf_swaps_flattened_ts_ev_fix_all_pct_deltas = chf_swaps_flattened_ts_ev_fix_all.pct_change(axis=1)
        chf_swaps_flattened_ts_ev_flt_all_pct_deltas = chf_swaps_flattened_ts_ev_flt_all.pct_change(axis=1)
        
        chf_swaps_flattened_ts_pv_fix_all_sum_mthly = chf_swaps_flattened_ts_pv_fix_all.sum(axis=0)
        chf_swaps_flattened_ts_pv_flt_all_sum_mthly = chf_swaps_flattened_ts_pv_flt_all.sum(axis=0)
        chf_swaps_flattened_ts_ev_fix_all_sum_mthly = chf_swaps_flattened_ts_ev_fix_all.sum(axis=0)
        chf_swaps_flattened_ts_ev_flt_all_sum_mthly = chf_swaps_flattened_ts_ev_flt_all.sum(axis=0)

        chf_swaps_flattened_ts_pv_fix_all_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_pv_fix_all_sum_mthly).transpose()
        chf_swaps_flattened_ts_pv_flt_all_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_pv_flt_all_sum_mthly).transpose()
        chf_swaps_flattened_ts_ev_fix_all_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_ev_fix_all_sum_mthly).transpose()
        chf_swaps_flattened_ts_ev_flt_all_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_ev_flt_all_sum_mthly).transpose()
        
        chf_swaps_flattened_ts_pv_fix_all_deltas_sum_mthly = chf_swaps_flattened_ts_pv_fix_all_deltas.sum(axis=0)
        chf_swaps_flattened_ts_pv_flt_all_deltas_sum_mthly = chf_swaps_flattened_ts_pv_flt_all_deltas.sum(axis=0)
        chf_swaps_flattened_ts_ev_fix_all_deltas_sum_mthly = chf_swaps_flattened_ts_ev_fix_all_deltas.sum(axis=0)
        chf_swaps_flattened_ts_ev_flt_all_deltas_sum_mthly = chf_swaps_flattened_ts_ev_flt_all_deltas.sum(axis=0)
        
        chf_swaps_flattened_ts_pv_fix_all_deltas_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_pv_fix_all_deltas_sum_mthly).transpose()
        chf_swaps_flattened_ts_pv_flt_all_deltas_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_pv_flt_all_deltas_sum_mthly).transpose()
        chf_swaps_flattened_ts_ev_fix_all_deltas_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_ev_fix_all_deltas_sum_mthly).transpose()
        chf_swaps_flattened_ts_ev_flt_all_deltas_sum_mthly = pd.DataFrame(chf_swaps_flattened_ts_ev_flt_all_deltas_sum_mthly).transpose()
        
        chf_swaps_flattened_ts_pv_fix_all_deltas_sum_swap = chf_swaps_flattened_ts_pv_fix_all_deltas.sum(axis=1)
        chf_swaps_flattened_ts_pv_flt_all_deltas_sum_swap = chf_swaps_flattened_ts_pv_flt_all_deltas.sum(axis=1)
        chf_swaps_flattened_ts_ev_fix_all_deltas_sum_swap = chf_swaps_flattened_ts_ev_fix_all_deltas.sum(axis=1)
        chf_swaps_flattened_ts_ev_flt_all_deltas_sum_swap = chf_swaps_flattened_ts_ev_flt_all_deltas.sum(axis=1)
    
        chf_swaps_flattened_ts_pv_fix_all_deltas_sum_swap = pd.DataFrame(chf_swaps_flattened_ts_pv_fix_all_deltas_sum_swap)  
        chf_swaps_flattened_ts_pv_flt_all_deltas_sum_swap = pd.DataFrame(chf_swaps_flattened_ts_pv_flt_all_deltas_sum_swap)  
        chf_swaps_flattened_ts_ev_fix_all_deltas_sum_swap = pd.DataFrame(chf_swaps_flattened_ts_ev_fix_all_deltas_sum_swap)  
        chf_swaps_flattened_ts_ev_flt_all_deltas_sum_swap = pd.DataFrame(chf_swaps_flattened_ts_ev_flt_all_deltas_sum_swap)  
    
    prev_date_fas = current_date_fas

data_val = pd.concat([chf_swaps_flattened_ts_pv_fix_all_deltas_sum_swap,chf_swaps_flattened_ts_pv_flt_all_deltas_sum_swap],axis=1)
#data_val = pd.concat([chf_swaps_flattened_ts_pv_fix_all_deltas['CLEAN_NPV_FIXED_2020-04-30']
#                     ,chf_swaps_flattened_ts_pv_flt_all_deltas['CLEAN_NPV_FLOAT_2020-04-30']],axis=1)
data_val = data_val.fillna(0)

plt.figure(num=3,figsize=(10, 7))
#plt.ticklabel_format(style='plain',axis='y') #anonymize
#plt.tick_params(labelleft=False)             #anonymize
plt.title("Fixed vs Float Leg Delta Value")
dend = shc.dendrogram(shc.linkage(data_val, method='ward'))

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_val)

plt.figure(num=4,figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Value")
plt.xlabel("Clean NPV Delta Fixed Leg")
plt.ylabel("Clean NPV Delta Float Leg")
#plt.ticklabel_format(style='plain',axis='both')    #anonymize
#plt.tick_params(labelleft=False,labelbottom=False) #anonymize
plt.scatter(data_val.iloc[:,0],data_val.iloc[:,1],c=cluster.labels_,cmap='rainbow')

data_ev2 = pd.concat([chf_swaps_flattened_ts_ev_fix_all_deltas_sum_swap,chf_swaps_flattened_ts_ev_flt_all_deltas_sum_swap],axis=1)
data_ev2 = data_ev2.fillna(0)

plt.figure(num=5,figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Event 2")
dend = shc.dendrogram(shc.linkage(data_ev2, method='ward'))

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_ev2)

plt.figure(num=6,figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Event 2")
plt.xlabel("Event 2 Delta Fixed Leg")
plt.ylabel("Event 2 Delta Float Leg")
plt.scatter(data_ev2.iloc[:,0],data_ev2.iloc[:,1],c=cluster.labels_,cmap='rainbow')

#################### Principal Component Analysis (PCA)

data_pca = chf_swaps_flattened_tf_all[['NOTIONAL'
                                     ,'INTERNAL_RATE_FIXED'
                                     ,'INTERNAL_RATE_FLOAT'
                                     ,'GCOC_RATE']]
x = data_pca.values
x = StandardScaler().fit_transform(x) # normalizing the features
np.mean(x),np.std(x)
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
data_pca_normalised = pd.DataFrame(x,columns=feat_cols)
pca_tf = PCA(n_components=2)
principalComponents_tf = pca_tf.fit_transform(x)
principal_tf_Df = pd.DataFrame(data=principalComponents_tf
                              ,columns=['principal component 1','principal component 2'])
print('Explained variation per principal component: {}'.format(pca_tf.explained_variance_ratio_))

plt.figure(num=7,figsize=(10, 7))
plt.xlabel('Principal Component - 1')
plt.ylabel('Principal Component - 2')
plt.title("Principal Component Analysis of Trade Facts")
targets = ['L','S']
targets_legend = ['Fixed Receiver Swaps','Fixed Payer Swaps']
colors = ['r','g']

for target, color in zip(targets,colors):
    indicesToKeep = chf_swaps_flattened_tf_all['L/S_FIXED'] == target
    principal_tf_Df = principal_tf_Df.set_index(indicesToKeep.index)
    plt.scatter(principal_tf_Df.loc[indicesToKeep,'principal component 1']
               ,principal_tf_Df.loc[indicesToKeep,'principal component 2'],c=color,s=2)

plt.legend(targets_legend)

#################### Time Series Analysis and Forecasting
# https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

# Data Preprocessing

data_fc1 = chf_swaps_flattened_ts_pv_fix_all_deltas_sum_mthly.copy()
data_fc2 = chf_swaps_flattened_ts_pv_flt_all_deltas_sum_mthly.copy()

data_fc1.columns = data_fc1.columns.str.replace("CLEAN_NPV_FIXED_", "")
data_fc2.columns = data_fc2.columns.str.replace("CLEAN_NPV_FLOAT_", "")

data_fc = data_fc1.add(data_fc2)
data_fc = pd.DataFrame(data_fc).transpose()
data_fc = data_fc.rename(columns={0:'CLEAN_NPV'})

# Visualizing Time Series Data

plt.figure(num=8,figsize=(10,7))
plt.plot(data_fc.index,data_fc,label="Clean NPV")
plt.title("All Swaps - Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Indexing with Time Series Data

data_fc.index = pd.DatetimeIndex(data=data_fc.index)
y = data_fc['CLEAN_NPV'].resample('M').mean()

# Decompose Time Series Data

pd.plotting.register_matplotlib_converters() #?????????????????????????????????

rcParams['figure.figsize']        = 10,7
#rcParams['axes.formatter.limits'] = -100,100 #anonymize
#rcParams['ytick.labelleft']       = False    #anonymize
decomposition = sm.tsa.seasonal_decompose(y,model='additive')
fig = decomposition.plot()
plt.show()

#pd.plotting.deregister_matplotlib_converters() #??????????????????????????????

# Time Series Forecasting with ARIMA

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal
                                             ,enforce_stationarity=False,enforce_invertibility=False)

            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except:
            continue

# Lowest AIC value: ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:128.25561634689154

# Fitting the ARIMA Model

mod = sm.tsa.statespace.SARIMAX(y,order=(1,1,1),seasonal_order=(1,1,0,12)
                                 ,enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(lags=2,figsize=(10,7))
plt.show()

# Validating Forecasts

plt.figure(num=11,figsize=(10,7))
pred = results.get_prediction(start=pd.to_datetime('2019-06-30'),dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax,label='One-step ahead Forecast',alpha=.7,figsize=(10, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1],color='k',alpha=.2)
ax.set_title("All Swaps - Deltas")
ax.set_xlabel('Date')
ax.set_ylabel('Value')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2019-06-30':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our Forecasts is {}'.format(round(mse,2)))
print('The Root Mean Squared Error of our Forecasts is {}'.format(round(np.sqrt(mse),2)))

# Producing and Visualizing Forecasts

plt.figure(num=12,figsize=(10,7))
pred_uc = results.get_forecast(steps=24)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed',figsize=(10,7))
pred_uc.predicted_mean.plot(ax=ax,label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1],color='k',alpha=.25)
ax.set_title("All Swaps - Deltas")
ax.set_xlabel('Date')
ax.set_ylabel('Value')
plt.legend()
plt.show()

# Time Series Modeling with Prophet

clean_pv_prophet = data_fc.copy()
clean_pv_prophet = clean_pv_prophet.reset_index()
clean_pv_prophet = clean_pv_prophet.rename(columns={'index':'ds','CLEAN_NPV':'y'})
clean_pv_model = Prophet(interval_width=0.95)
clean_pv_model.fit(clean_pv_prophet)
clean_pv_forecast = clean_pv_model.make_future_dataframe(periods=36,freq='M')
clean_pv_forecast = clean_pv_model.predict(clean_pv_forecast)

# Plot 13
clean_pv_model.plot(clean_pv_forecast,xlabel='Date',ylabel='Value',figsize=(10,7))
plt.title('All Swaps - Deltas')

# Plot 14
components_fig = clean_pv_model.plot_components(clean_pv_forecast,figsize=(10,7))
axes_proph = components_fig.get_axes()
axes_proph[0].set_xlabel('Year')
axes_proph[0].set_title('All Swaps - Deltas')

# Plot 15
clean_pv_cv = cross_validation(clean_pv_model,horizon = '180 days')
plot_cross_validation_metric(clean_pv_cv,metric='mape',figsize=(10,7))
plt.title('All Swaps - Deltas')

#################### Time Series Prediction with a Neural Network
# https://www.liip.ch/en/blog/time-series-prediction-a-short-comparison-of-best-practices
# https://github.com/plotti/timeseries_demo/blob/master/Neural%20Network.ipynb

clean_pv_nn = data_fc.copy()

# Split a Univariate Dataset into Train/Test Sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# Transform List into Supervised Learning Format
def series_to_supervised(data, n_in, n_out=1):
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# Root Mean Squared Error or RMSE
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# Fit a Model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch = config
    #[24, 500, 100, 100]
    # prepare data
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
	# define model
	model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# Forecast with a Pre-Fit Model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _ = config
	# prepare data
	x_input = array(history[-n_input:]).reshape(1, n_input)
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# Walk-Forward Validation for Univariate Data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return [error,predictions]

# Repeat Evaluation of a Config
def repeat_evaluate(data, config, n_test, n_repeats=5):
	# fit and evaluate the model n times
    scores = []
    predictions = []
    for _ in range(n_repeats):
        tmp_scores, tmp_predictions = walk_forward_validation(data, n_test, config)
        scores.append(tmp_scores)
        predictions.append(tmp_predictions)
    return [scores,predictions]

# Summarize Model Performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	plt.boxplot(scores)
	plt.show()

def find_best_solution(series,name,method="nn"):
    data = series.values
    n_test = int(len(data)*0.2) # percentage used for test
    #config = [24, 500, 100, 100]
    config = [12, 500, 100, 5]
    scores,predictions = repeat_evaluate(data, config, n_test)
    summarize_scores('mlp', scores)
    #plt.savefig("scores_%s_%s.png" % (name,method))
    plt.show()
    train,test = train_test_split(data,n_test)
    prediction = pd.DataFrame(list(train.flatten())+np.array(predictions[0]).flatten().tolist())
    ax = pd.DataFrame(data).plot(label="Original") # main data
    prediction.plot(ax=ax, alpha=.7, figsize=(10,7))
    #plt.savefig("pred_%s_%s.png" %(name,method))
    plt.show()

find_best_solution(clean_pv_nn,"CLEAN_PV")

