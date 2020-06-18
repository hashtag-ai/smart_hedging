# Smart Hedging Main Routine

import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import glob

wdir = os.getcwd()
datadir = os.path.join(wdir,"data\TPC")

fas133_data = "/fas133_data_*.xlsx"
full_path_fas133_data = datadir+fas133_data

#all_fas_files = glob.glob(full_path_fas133_data)
all_fas_files = [datadir+"/fas133_data_20200331.xlsx",datadir+"/fas133_data_20200430.xlsx"]

ind = 0

for filename in all_fas_files:
    ind = ind + 1
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
    
    chf_swaps_flattened = chf_swaps_flattened.drop(columns=['CCY_FIXED'
                                                           ,'CCY_FLOAT'
                                                           ,'RATE_TYPE_FIXED'
                                                           ,'RATE_REF_FIXED'
                                                           ,'COB Date_FLOAT'
                                                           ,'GCOC_RATE_FLOAT'
                                                           ,'Start Date_FLOAT'
                                                           ,'End Date_FLOAT'
                                                           ,'RATE_TYPE_FLOAT'
                                                           ,'Notional_FLOAT'
                                                           ],axis=1)
    
    current_date = chf_swaps_flattened['COB Date_FIXED'].dt.date.iloc[0]
    
    if ind == 1:
        chf_swaps_flattened_all = chf_swaps_flattened
        chf_swaps_flattened_all = chf_swaps_flattened_all.add_suffix('_').add_suffix(chf_swaps_flattened['COB Date_FIXED'].dt.date.iloc[0])
        chf_swaps_flattened_all = chf_swaps_flattened_all.rename(columns={chf_swaps_flattened_all.columns[1]:'Deal No'})
        chf_swaps_flattened_all = chf_swaps_flattened_all.drop(columns=[chf_swaps_flattened_all.columns[0]],axis=1)
    else:
        
        chf_swaps_flattened = chf_swaps_flattened.add_suffix('_').add_suffix(chf_swaps_flattened['COB Date_FIXED'].dt.date.iloc[0])
        chf_swaps_flattened = chf_swaps_flattened.rename(columns={chf_swaps_flattened.columns[1]:'Deal No'})
        chf_swaps_flattened = chf_swaps_flattened.drop(columns=[chf_swaps_flattened.columns[0]],axis=1)
        chf_swaps_flattened_all = pd.merge(chf_swaps_flattened_all,chf_swaps_flattened
                                          ,how='outer',on='Deal No'
                                          ,left_index=False,right_index=False,sort=False
                                          ,suffixes=(False,False),copy=True,indicator=False
                                          ,validate="1:1")
        
        target_column_pv_fix = "CLEAN_NPV_FIXED_DELTA_" + str(current_date)
        source_column_new    = "CLEAN_NPV_FIXED_" + str(current_date)
        source_column_old    = "CLEAN_NPV_FIXED_" + str(prev_date)
        chf_swaps_flattened_all = chf_swaps_flattened_all.assign(target_column_pv_fix
                                                                 = chf_swaps_flattened_all[source_column_new]
                                                                 - chf_swaps_flattened_all[source_column_old])
        chf_swaps_flattened_all = chf_swaps_flattened_all.rename(columns={'target_column_pv_fix':target_column_pv_fix})
        
        target_column_pv_flt = "CLEAN_NPV_FLOAT_DELTA_" + str(current_date)
        source_column_new    = "CLEAN_NPV_FLOAT_" + str(current_date)
        source_column_old    = "CLEAN_NPV_FLOAT_" + str(prev_date)
        chf_swaps_flattened_all = chf_swaps_flattened_all.assign(target_column_pv_flt
                                                                 = chf_swaps_flattened_all[source_column_new]
                                                                 - chf_swaps_flattened_all[source_column_old])
        chf_swaps_flattened_all = chf_swaps_flattened_all.rename(columns={'target_column_pv_flt':target_column_pv_flt})
    
        target_column_02_fix = "EVENT_02_FIXED_DELTA_" + str(current_date)
        source_column_new    = "EVENT_02_FIXED_" + str(current_date)
        source_column_old    = "EVENT_02_FIXED_" + str(prev_date)
        chf_swaps_flattened_all = chf_swaps_flattened_all.assign(target_column_02_fix
                                                                 = chf_swaps_flattened_all[source_column_new]
                                                                 - chf_swaps_flattened_all[source_column_old])
        chf_swaps_flattened_all = chf_swaps_flattened_all.rename(columns={'target_column_02_fix':target_column_02_fix})
    
        target_column_02_flt = "EVENT_02_FLOAT_DELTA_" + str(current_date)
        source_column_new    = "EVENT_02_FLOAT_" + str(current_date)
        source_column_old    = "EVENT_02_FLOAT_" + str(prev_date)
        chf_swaps_flattened_all = chf_swaps_flattened_all.assign(target_column_02_flt
                                                                 = chf_swaps_flattened_all[source_column_new]
                                                                 - chf_swaps_flattened_all[source_column_old])
        chf_swaps_flattened_all = chf_swaps_flattened_all.rename(columns={'target_column_02_flt':target_column_02_flt})
    
    prev_date = current_date

data_val = chf_swaps_flattened_all[[target_column_pv_fix,target_column_pv_flt]].copy()
data_val = data_val.fillna(0)

plt.figure(figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Value")
dend = shc.dendrogram(shc.linkage(data_val, method='ward'))

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_val)

plt.figure(figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Value")
plt.xlabel("Clean NPV Delta Fixed Leg")
plt.ylabel("Clean NPV Delta Float Leg")
plt.scatter(data_val.iloc[:,0],data_val.iloc[:,1],c=cluster.labels_,cmap='rainbow')

data_ev2 = chf_swaps_flattened_all[[target_column_02_fix,target_column_02_flt]].copy()
data_ev2 = data_ev2.fillna(0)

plt.figure(figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Event 2")
dend = shc.dendrogram(shc.linkage(data_ev2, method='ward'))

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_ev2)

plt.figure(figsize=(10, 7))
plt.title("Fixed vs Float Leg Delta Event 2")
plt.xlabel("Event 2 Delta Fixed Leg")
plt.ylabel("Event 2 Delta Float Leg")
plt.scatter(data_ev2.iloc[:,0],data_ev2.iloc[:,1],c=cluster.labels_,cmap='rainbow')

