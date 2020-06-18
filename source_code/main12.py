# Smart Hedging Main Routine

import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import numpy as np

wdir = os.getcwd()
datadir = os.path.join(wdir,"data\TPC")

fas133_data_filename = "/fas133_data_20200430.xlsx"
full_path_fas133_data = datadir+fas133_data_filename

fas133_data = pd.read_excel(full_path_fas133_data
                           ,sheet_name = 'Sheet1'
                           ,usecols=[
#                                    'COB Date',
                                     'Deal No','CCY','L/S'
                                    ,'Start Date','End Date','FREQUENCY'
                                    ,'RATE_TYPE','RATE_REF','INTERNAL_RATE','GCOC_RATE'
                                    ,'Notional'
                                    ,'CLEAN_NPV','EVENT_02'
                                    ])

chf_swaps = fas133_data.loc[(fas133_data['CCY'] == "CHF")
                          & ((((fas133_data['RATE_REF'] == "3MLIBOR")   |
                               (fas133_data['RATE_REF'] == "6MLIBOR"))  &
                               (fas133_data['RATE_TYPE'] == "FLOAT"))   |
                               (fas133_data['RATE_TYPE'] == "FIXED"))
                           ]

chf_swaps = chf_swaps.drop(columns=['CCY'],axis=1)

chf_swaps_fixed_legs = chf_swaps.loc[chf_swaps['RATE_TYPE'] == "FIXED"]
chf_swaps_float_legs = chf_swaps.loc[chf_swaps['RATE_TYPE'] == "FLOAT"]

chf_swaps_fixed_legs = chf_swaps_fixed_legs.add_suffix('_FIXED')
chf_swaps_float_legs = chf_swaps_float_legs.add_suffix('_FLOAT')

chf_swaps_fixed_legs = chf_swaps_fixed_legs.rename(columns={'Deal No_FIXED': 'Deal No'})
chf_swaps_float_legs = chf_swaps_float_legs.rename(columns={'Deal No_FLOAT': 'Deal No'})

chf_swaps_flattened = pd.merge(chf_swaps_fixed_legs,chf_swaps_float_legs
                              ,how='inner',on='Deal No'
                              ,left_index=False,right_index=False,sort=False
                              ,suffixes=('_x', '_y'),copy=True,indicator=False
                              ,validate="1:1")

chf_swaps_flattened = chf_swaps_flattened.drop(columns=['RATE_TYPE_FIXED'
                                                      ,'RATE_REF_FIXED'
                                                      ,'GCOC_RATE_FLOAT'
                                                      ,'Start Date_FLOAT'
                                                      ,'End Date_FLOAT'
                                                      ,'RATE_TYPE_FLOAT'
                                                      ,'Notional_FLOAT'
                                                      ],axis=1)

chf_swaps_flattened = chf_swaps_flattened.drop(columns=['Deal No'],axis=1)

data = chf_swaps_flattened.iloc[:,[7,13]].values
data = np.nan_to_num(data)
#data = chf_swaps_flattened.fillna(0)

plt.figure(figsize=(10, 7))
plt.title("Fixed vs Float Leg Value")
dend = shc.dendrogram(shc.linkage(data, method='ward'))

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.title("Fixed vs Float Leg Value")
plt.xlabel("Clean NPV Fixed Leg")
plt.ylabel("Clean NPV Float Leg")
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

