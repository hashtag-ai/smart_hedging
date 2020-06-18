# Smart Hedging Main Routine

import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from operator import attrgetter

buckets = pd.DataFrame([['1M' ,-100,  1,    0,   30]
                       ,['2M' ,   2,  2,   31,   60]
                       ,['3M' ,   3,  3,   61,   90]
                       ,['6M' ,   4,  6,   91,  180]
                       ,['9M' ,   7,  9,  181,  270]
                       ,['1Y' ,  10, 12,  271,  360]
                       ,['15M',  13, 15,  361,  450]
                       ,['18M',  16, 18,  451,  540]
                       ,['21M',  19, 21,  541,  630]
                       ,['2Y' ,  22, 24,  631,  720]
                       ,['3Y' ,  25, 36,  721, 1080]
                       ,['4Y' ,  37, 48, 1081, 1440]
                       ,['5Y' ,  49, 60, 1441, 1800]
                       ,['6Y' ,  61, 72, 1801, 2160]
                       ,['7Y' ,  73, 84, 2161, 2520]
                       ,['8Y' ,  85, 96, 2521, 2880]
                       ,['9Y' ,  97,108, 2881, 3240]
                       ,['10Y', 109,120, 3241, 3600]
                       ,['11Y', 121,132, 3601, 3960]
                       ,['12Y', 133,144, 3961, 4320]
                       ,['13Y', 145,156, 4321, 4680]
                       ,['14Y', 157,168, 4681, 5040]
                       ,['15Y', 169,180, 5041, 5400]
                       ,['20Y', 181,240, 5401, 7200]
                       ,['25Y', 241,300, 7201, 9000]
                       ,['30Y', 301,360, 9001,10800]
                       ,['50Y', 361,600,10801,20000]]
                      ,columns=['BUCKET_LABEL','MONTH_FROM','MONTH_TO','DAY_FROM','DAY_TO'])

wdir = os.getcwd()
datadir_yc = os.path.join(wdir,"data")
datadir_fas = os.path.join(wdir,"data\TPC")

file_chflibor = "/chflibor_monthly.xlsx"
full_path_chflibor = datadir_yc+file_chflibor

chflibor_temp = pd.read_excel(full_path_chflibor
                        ,sheet_name = 'CHFLIBOR'
                        ,usecols=['VAL_DATE','DAYS_FWD','ZERO_RATE'])

# Set Filter
#chflibor = pd.DataFrame(chflibor_temp[(chflibor_temp['VAL_DATE'].dt.date >= pd.to_datetime("2000-12-31"))
#                                    & (chflibor_temp['VAL_DATE'].dt.month == 12)])
chflibor = pd.DataFrame(chflibor_temp[(chflibor_temp['VAL_DATE'].dt.date >= pd.to_datetime("2017-12-31"))])
#chflibor = pd.DataFrame(chflibor_temp)

chflibor['VAL_DATE'] = chflibor['VAL_DATE'].dt.date

dates_yc = chflibor['VAL_DATE'].unique()
dfwd_max = chflibor['DAYS_FWD'].max()
chflibor_all = pd.DataFrame(index=range(1,dfwd_max+1),columns=dates_yc)
#chflibor_all = pd.DataFrame(index=range(0,dfwd_max+1),columns=dates_yc) # in case of all Libor rates since 2003
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

#plt.figure(num=1)
#plt.scatter(chflibor_all_ipl.index,chflibor_all_ipl['ZERO_RATE_2020-04-30'],label="ZERO_RATE_2020-04-30",s=2)
#plt.title("CHF Libor Interpolated")
#plt.xlabel("Days Forward")
#plt.ylabel("Zero Rate")
#plt.legend()
#plt.show()

### 3D Plot

chflibor_all_ipl_3d = chflibor_all_ipl.copy()
chflibor_all_ipl_3d.columns = chflibor_all_ipl_3d.columns.str.replace("ZERO_RATE_", "")

x = chflibor_all_ipl_3d.index
y = np.arange(len(chflibor_all_ipl_3d.columns))
X,Y = np.meshgrid(x,y)
Z = chflibor_all_ipl_3d

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,np.transpose(Z),rstride=100,cstride=100,cmap=cm.ocean,antialiased=True,shade=False)

ax.set_xlabel('Days Forward')
ax.set_ylabel('Date')
ax.set_yticklabels(['0','2017-12-31','2018-05-31','2018-10-31','2019-03-31','2019-08-31','2020-01-31'],minor=False)
#ax.set_yticks([0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204])
#ax.set_yticklabels(['2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014'
#                   ,'2015','2016','2017','2018','2019','2020','2021'],minor=False)
ax.set_zlabel('Zero Rate')
plt.tick_params(labelleft=True)

plt.show()

###

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

chflibor_all_ipl_bucket = chflibor_all_ipl.copy()
br = 0
for br, row in chflibor_all_ipl_bucket.iterrows():
    dfwd = chflibor_all_ipl_bucket.index[br-1]
    bucket = buckets['BUCKET_LABEL'].loc[(buckets['DAY_FROM'] <= dfwd) & (buckets['DAY_TO'] >= dfwd)]
    bucket_val = bucket.iloc[0]
    chflibor_all_ipl_bucket.at[br,'BUCKET_LABEL'] = bucket_val

chflibor_all_ipl_bucket_avg = pd.DataFrame(buckets['BUCKET_LABEL']).copy()
chflibor_all_ipl_bucket_avg = chflibor_all_ipl_bucket_avg.set_index('BUCKET_LABEL')
avg_calc_temp = pd.DataFrame(chflibor_all_ipl_bucket.groupby('BUCKET_LABEL').mean())
chflibor_all_ipl_bucket_avg = pd.concat([chflibor_all_ipl_bucket_avg,avg_calc_temp],axis=1,sort=False)

chflibor_all_ipl_deltas_bucket = chflibor_all_ipl_deltas.copy()
br = 0
for br, row in chflibor_all_ipl_deltas_bucket.iterrows():
    dfwd = chflibor_all_ipl_deltas_bucket.index[br-1]
    bucket = buckets['BUCKET_LABEL'].loc[(buckets['DAY_FROM'] <= dfwd) & (buckets['DAY_TO'] >= dfwd)]
    bucket_val = bucket.iloc[0]
    chflibor_all_ipl_deltas_bucket.at[br,'BUCKET_LABEL'] = bucket_val

chflibor_all_ipl_deltas_bucket_avg = pd.DataFrame(buckets['BUCKET_LABEL']).copy()
chflibor_all_ipl_deltas_bucket_avg = chflibor_all_ipl_deltas_bucket_avg.set_index('BUCKET_LABEL')
avg_calc_temp = pd.DataFrame(chflibor_all_ipl_deltas_bucket.groupby('BUCKET_LABEL').mean())
chflibor_all_ipl_deltas_bucket_avg = pd.concat([chflibor_all_ipl_deltas_bucket_avg,avg_calc_temp],axis=1,sort=False)

fas133_data = "/fas133_data_*.xlsx"
full_path_fas133_data = datadir_fas+fas133_data

all_fas_files = glob.glob(full_path_fas133_data)
all_fas_files.sort()
#all_fas_files = [datadir_fas+"/fas133_data_20200228.xlsx",datadir_fas+"/fas133_data_20200331.xlsx"
#                ,datadir_fas+"/fas133_data_20200430.xlsx"]
#all_fas_files = [datadir_fas+"/fas133_data_20190830.xlsx"
#                ,datadir_fas+"/fas133_data_20190930.xlsx",datadir_fas+"/fas133_data_20191029.xlsx"
#                ,datadir_fas+"/fas133_data_20191129.xlsx",datadir_fas+"/fas133_data_20191231.xlsx"]

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
    
    chf_swaps_flattened_ts_mat_mths = chf_swaps_flattened_tf[['DEAL_NO','COB_DATE_FIRST','END_DATE']].copy()
    chf_swaps_flattened_ts_mat_mths = chf_swaps_flattened_ts_mat_mths.assign(MONTHS_DIF
                                                                   = (chf_swaps_flattened_ts_mat_mths.END_DATE.dt.to_period('M')
                                                                    - chf_swaps_flattened_ts_mat_mths.COB_DATE_FIRST.dt.to_period('M')).apply(attrgetter('n')))
    
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
    target_column_pv_fix   = "CLEAN_NPV_FIXED_" + str(current_date_fas)
    target_column_pv_flt   = "CLEAN_NPV_FLOAT_" + str(current_date_fas)
    target_column_ev_fix   = "EVENT_02_FIXED_"  + str(current_date_fas)
    target_column_ev_flt   = "EVENT_02_FLOAT_"  + str(current_date_fas)
    target_column_mat_mths = "MAT_MTHS_"        + str(current_date_fas)
    target_column_bucket   = "BUCKET_LABEL_"    + str(current_date_fas)

    chf_swaps_flattened_ts_pv_fix = chf_swaps_flattened_ts_pv_fix.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'CLEAN_NPV_FIXED':target_column_pv_fix})
    chf_swaps_flattened_ts_pv_flt = chf_swaps_flattened_ts_pv_flt.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'CLEAN_NPV_FLOAT':target_column_pv_flt})
    chf_swaps_flattened_ts_ev_fix = chf_swaps_flattened_ts_ev_fix.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'EVENT_02_FIXED':target_column_ev_fix})
    chf_swaps_flattened_ts_ev_flt = chf_swaps_flattened_ts_ev_flt.rename(columns={'Deal No':'DEAL_NO'
                                                                                 ,'EVENT_02_FLOAT':target_column_ev_flt})
   
    chf_swaps_flattened_ts_mat_mths = chf_swaps_flattened_ts_mat_mths.rename(columns={'MONTHS_DIF':target_column_mat_mths})
    chf_swaps_flattened_ts_mat_mths = chf_swaps_flattened_ts_mat_mths.drop(columns=['COB_DATE_FIRST','END_DATE'])
    
    chf_swaps_flattened_ts_bucket = chf_swaps_flattened_ts_mat_mths.copy()
    chf_swaps_flattened_ts_bucket[target_column_bucket] = ''
    bu = 0
    for bu, row in chf_swaps_flattened_ts_bucket.iterrows():
        mat_mths = row[1]
        bucket = buckets['BUCKET_LABEL'].loc[(buckets['MONTH_FROM'] <= mat_mths) & (buckets['MONTH_TO'] >= mat_mths)]
        bucket_val = bucket.iloc[0]
        chf_swaps_flattened_ts_bucket.at[bu,target_column_bucket] = bucket_val
    
    ### Below not functioning ###
    # chf_swaps_flattened_ts_bucket[target_column_bucket] = buckets['BUCKET_LABEL'].loc[(buckets['MONTH_FROM'] <= chf_swaps_flattened_ts_bucket[target_column_mat_mths])
    #                                                                                 & (buckets['MONTH_TO']   >= chf_swaps_flattened_ts_bucket[target_column_mat_mths])]

    chf_swaps_flattened_ts_bucket = chf_swaps_flattened_ts_bucket.drop(columns=[target_column_mat_mths])
    
    if ind == 1:
        chf_swaps_flattened_tf_all          = chf_swaps_flattened_tf.copy()
        chf_swaps_flattened_ts_pv_fix_all   = chf_swaps_flattened_ts_pv_fix.copy()
        chf_swaps_flattened_ts_pv_flt_all   = chf_swaps_flattened_ts_pv_flt.copy()
        chf_swaps_flattened_ts_ev_fix_all   = chf_swaps_flattened_ts_ev_fix.copy()
        chf_swaps_flattened_ts_ev_flt_all   = chf_swaps_flattened_ts_ev_flt.copy()
        chf_swaps_flattened_ts_mat_mths_all = chf_swaps_flattened_ts_mat_mths.copy()
        chf_swaps_flattened_ts_bucket_all   = chf_swaps_flattened_ts_bucket.copy()
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
        chf_swaps_flattened_ts_mat_mths_all = pd.merge(chf_swaps_flattened_ts_mat_mths_all,chf_swaps_flattened_ts_mat_mths
                                                    ,how='outer',on='DEAL_NO'
                                                    ,left_index=False,right_index=False,sort=False
                                                    ,suffixes=(False,False),copy=True,indicator=False
                                                    ,validate="1:1")
        chf_swaps_flattened_ts_bucket_all = pd.merge(chf_swaps_flattened_ts_bucket_all,chf_swaps_flattened_ts_bucket
                                                    ,how='outer',on='DEAL_NO'
                                                    ,left_index=False,right_index=False,sort=False
                                                    ,suffixes=(False,False),copy=True,indicator=False
                                                    ,validate="1:1")
    
    prev_date_fas = current_date_fas

chf_swaps_flattened_ts_pv_fix_all   = chf_swaps_flattened_ts_pv_fix_all.set_index('DEAL_NO')
chf_swaps_flattened_ts_pv_flt_all   = chf_swaps_flattened_ts_pv_flt_all.set_index('DEAL_NO')
chf_swaps_flattened_ts_ev_fix_all   = chf_swaps_flattened_ts_ev_fix_all.set_index('DEAL_NO')
chf_swaps_flattened_ts_ev_flt_all   = chf_swaps_flattened_ts_ev_flt_all.set_index('DEAL_NO')
chf_swaps_flattened_ts_mat_mths_all = chf_swaps_flattened_ts_mat_mths_all.set_index('DEAL_NO')
chf_swaps_flattened_ts_bucket_all   = chf_swaps_flattened_ts_bucket_all.set_index('DEAL_NO')

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

### Bucket Movement Analysis
### PVs

df1 = chf_swaps_flattened_ts_pv_fix_all_deltas.copy()
df2 = chf_swaps_flattened_ts_pv_flt_all_deltas.copy()
df1.columns = df1.columns.str.replace("CLEAN_NPV_FIXED_", "CLEAN_NPV_")
df2.columns = df2.columns.str.replace("CLEAN_NPV_FLOAT_", "CLEAN_NPV_")
chf_swaps_flattened_ts_pv_all_deltas = df1.add(df2)

# Scale PVs

chf_swaps_flattened_ts_bucket_all_scld = chf_swaps_flattened_ts_bucket_all.copy()
chf_swaps_flattened_ts_bucket_all_scld.columns = chf_swaps_flattened_ts_bucket_all_scld.columns.str.replace("BUCKET_LABEL_", "ZERO_RATE_")

scc = 0
scr = 0
for scc,column in chflibor_all_ipl_deltas_bucket_avg.iteritems():
    pos = chflibor_all_ipl_deltas_bucket_avg.columns.get_loc(scc)
    target_column_yc = chf_swaps_flattened_ts_bucket_all_scld.columns[pos]
    for scr,row in chflibor_all_ipl_deltas_bucket_avg.iterrows():
        chf_swaps_flattened_ts_bucket_all_scld.loc[(chf_swaps_flattened_ts_bucket_all_scld[target_column_yc] == scr),target_column_yc] = row[pos]

df7 = chf_swaps_flattened_ts_pv_all_deltas.copy()
df8 = chf_swaps_flattened_ts_bucket_all_scld.copy()
df7.columns = df7.columns.str.replace("CLEAN_NPV_", "SCALED_CLEAN_NPV_")
df8.columns = df8.columns.str.replace("ZERO_RATE_", "SCALED_CLEAN_NPV_")

df8 = df8.multiply(10000)
chf_swaps_flattened_ts_pv_all_deltas_scld = df7.divide(df8)

### EVs

df3 = chf_swaps_flattened_ts_ev_fix_all_deltas.copy()
df4 = chf_swaps_flattened_ts_ev_flt_all_deltas.copy()
df3.columns = df3.columns.str.replace("EVENT_02_FIXED_", "EVENT_02_")
df4.columns = df4.columns.str.replace("EVENT_02_FLOAT_", "EVENT_02_")
chf_swaps_flattened_ts_ev_all_deltas = df3.add(df4)

# Scale EVs

df9  = chf_swaps_flattened_ts_ev_all_deltas.copy()
df10 = chf_swaps_flattened_ts_bucket_all_scld.copy()
df9.columns  = df9.columns.str.replace("EVENT_02_", "SCALED_EVENT_02_")
df10.columns = df10.columns.str.replace("ZERO_RATE_", "SCALED_EVENT_02_")

df10 = df10.multiply(10000)
chf_swaps_flattened_ts_ev_all_deltas_scld = df9.divide(df10)

### All

df5 = chf_swaps_flattened_ts_pv_all_deltas.copy()
df6 = chf_swaps_flattened_ts_ev_all_deltas.copy()
df5.columns = df5.columns.str.replace("CLEAN_NPV_", "ALL_")
df6.columns = df6.columns.str.replace("EVENT_02_", "ALL_")
chf_swaps_flattened_ts_all_deltas = df5.add(df6)

# Scale All

df11 = chf_swaps_flattened_ts_all_deltas.copy()
df12 = chf_swaps_flattened_ts_bucket_all_scld.copy()
df11.columns = df11.columns.str.replace("ALL_", "SCALED_ALL_")
df12.columns = df12.columns.str.replace("ZERO_RATE_", "SCALED_ALL_")

df12 = df12.multiply(10000)
chf_swaps_flattened_ts_all_deltas_scld = df11.divide(df12)

###

pay_swaps = chf_swaps_flattened_tf_all.loc[(chf_swaps_flattened_tf_all['L/S_FIXED'] == "S")]
rec_swaps = chf_swaps_flattened_tf_all.loc[(chf_swaps_flattened_tf_all['L/S_FIXED'] == "L")]

#dates = ['2020-02-28','2020-03-31','2020-04-30']
#dates = ['2019-06-28','2019-07-31','2019-08-30','2019-09-30','2019-10-29','2019-11-29'
#        ,'2019-12-31','2020-01-31','2020-02-28','2020-03-31','2020-04-30']
#dates = ['2019-06-28','2019-07-31','2019-08-30','2019-09-30','2019-10-29']
dates = ['2019-09-30','2019-10-29','2019-11-29','2019-12-31']
#dates = df1.columns.str.replace("CLEAN_NPV_", "").values
#dates = df3.columns.str.replace("EVENT_02_", "").values
#dates = df5.columns.str.replace("ALL_", "").values
#dates = df7.columns.str.replace("SCALED_CLEAN_NPV_", "").values
#dates = df9.columns.str.replace("SCALED_EVENT_02_", "").values
#dates = df11.columns.str.replace("SCALED_ALL_", "").values
prev_dt_col_bs = ""
prev_zr_col    = ""
dtc = 0
for dt in dates:
    dtc += 1
    dt_col_bu = 'BUCKET_LABEL_' + dt
    
    #dt_col_bs = 'BUCKET_SUM_' + dt
    dt_col_bs = 'SCALED_BUCKET_SUM_' + dt
    
    #mytitle = "Delta Value per Bucket - All Swaps"
    #mytitle = "Delta Value per Bucket - Payer Swaps"
    #mytitle = "Delta Value per Bucket - Receiver Swaps"
    #dt_col  = 'CLEAN_NPV_' + dt
    #pos     = df1.columns.get_loc(dt_col)
    #df_sum  = pd.concat([chf_swaps_flattened_ts_pv_all_deltas[dt_col],chf_swaps_flattened_ts_bucket_all[dt_col_bu]],axis=1).copy()
   
    #mytitle = "Scaled Delta Value per Bucket - All Swaps"
    #mytitle = "Scaled Delta Value per Bucket - Payer Swaps"
    #mytitle = "Scaled Delta Value per Bucket - Receiver Swaps"
    mytitle = "Scaled Delta Value per Bucket - Perfect Hedge"
    dt_col  = 'SCALED_CLEAN_NPV_' + dt
    pos     = df7.columns.get_loc(dt_col)
    df_sum  = pd.concat([chf_swaps_flattened_ts_pv_all_deltas_scld[dt_col],chf_swaps_flattened_ts_bucket_all[dt_col_bu]],axis=1).copy()
    
    #mytitle = "Delta Event 02 per Bucket - All Swaps"
    #mytitle = "Delta Event 02 per Bucket - Payer Swaps"
    #mytitle = "Delta Event 02 per Bucket - Receiver Swaps"
    #dt_col  = 'EVENT_02_' + dt
    #pos     = df3.columns.get_loc(dt_col)
    #df_sum  = pd.concat([chf_swaps_flattened_ts_ev_all_deltas[dt_col],chf_swaps_flattened_ts_bucket_all[dt_col_bu]],axis=1).copy()
    
    #mytitle = "Scaled Delta Event 02 per Bucket - All Swaps"
    #mytitle = "Scaled Delta Event 02 per Bucket - Payer Swaps"
    #mytitle = "Scaled Delta Event 02 per Bucket - Receiver Swaps"
    #dt_col  = 'SCALED_EVENT_02_' + dt
    #pos     = df9.columns.get_loc(dt_col)
    #df_sum  = pd.concat([chf_swaps_flattened_ts_ev_all_deltas_scld[dt_col],chf_swaps_flattened_ts_bucket_all[dt_col_bu]],axis=1).copy()
    
    #mytitle = "Delta Value + Delta Event 02 per Bucket - All Swaps"
    #mytitle = "Delta Value + Delta Event 02 per Bucket - Payer Swaps"
    #mytitle = "Delta Value + Delta Event 02 per Bucket - Receiver Swaps"
    #dt_col  = 'ALL_' + dt
    #pos     = df5.columns.get_loc(dt_col)
    #df_sum  = pd.concat([chf_swaps_flattened_ts_all_deltas[dt_col],chf_swaps_flattened_ts_bucket_all[dt_col_bu]],axis=1).copy()
    
    #mytitle = "Scaled Delta Value + Scaled Delta Event 02 per Bucket - All Swaps"
    #mytitle = "Scaled Delta Value + Scaled Delta Event 02 per Bucket - Payer Swaps"
    #mytitle = "Scaled Delta Value + Scaled Delta Event 02 per Bucket - Receiver Swaps"
    #dt_col  = 'SCALED_ALL_' + dt
    #pos     = df11.columns.get_loc(dt_col)
    #df_sum  = pd.concat([chf_swaps_flattened_ts_all_deltas_scld[dt_col],chf_swaps_flattened_ts_bucket_all[dt_col_bu]],axis=1).copy()
    
    # Set Filter: Hedged -> All Receiver Swaps and Needed Payer Swaps for the Perfect Hedge
    df_sum_pay = df_sum.loc[df_sum.index.isin(pay_swaps.index)]
    df_sum_rec = df_sum.loc[df_sum.index.isin(rec_swaps.index)]
    df_sum_pay[dt_col] = df_sum_pay[dt_col].fillna(0)
    df_sum_rec[dt_col] = df_sum_rec[dt_col].fillna(0)
    amt_hedge = 0
    df_sum_pay_bu_sel = df_sum_pay.copy()
    df_sum_pay_bu_sel = df_sum_pay_bu_sel.drop(df_sum_pay_bu_sel.index,axis=0)
    for bh,row1 in buckets.iterrows():
        bh = row1[0]
        amt_hedge = df_sum_rec.loc[(df_sum_rec[dt_col_bu]==bh)].sum().values[0]
        df_sum_pay_bu = df_sum_pay.loc[(df_sum_pay[dt_col_bu]==bh)]
        prev_amt_hedge = amt_hedge
        for he,row2 in df_sum_pay_bu.iterrows():
            amt_hedge = amt_hedge + row2[0]
            if abs(prev_amt_hedge) < abs(amt_hedge):
                break
            prev_amt_hedge = amt_hedge
            df_sum_pay_bu_sel = df_sum_pay_bu_sel.append(df_sum_pay_bu.loc[df_sum_pay_bu.index==he])
    df_sum  = df_sum_rec.append(df_sum_pay_bu_sel)
    
    # Set Filter: All Swaps, Payer Swaps or Receiver Swaps Only
    #df_sum = df_sum_pay.copy()
    #df_sum = df_sum_rec.copy()
    
    chf_swaps_flattened_ts_bucket_sum = pd.DataFrame(buckets['BUCKET_LABEL']).copy()
    chf_swaps_flattened_ts_bucket_sum = chf_swaps_flattened_ts_bucket_sum.set_index('BUCKET_LABEL')
    chf_swaps_flattened_ts_bucket_sum[dt_col_bs] = pd.DataFrame(df_sum.groupby(dt_col_bu).sum())
    chf_swaps_flattened_ts_bucket_sum = chf_swaps_flattened_ts_bucket_sum.fillna(0)

    plt.figure(num=dt,figsize=(15,7))
    plt.bar(chf_swaps_flattened_ts_bucket_sum.index,chf_swaps_flattened_ts_bucket_sum[dt_col_bs]
           ,label=dt_col_bs,width=0.3,color='royalblue',align='edge')

    #if dtc > 2:
    if dtc > 1:
        plt.bar(chf_swaps_flattened_ts_bucket_sum.index,chf_swaps_flattened_ts_bucket_sum_all[prev_dt_col_bs]
               ,label=prev_dt_col_bs,width=-0.3,color='lightgrey',align='edge')
    
    plt.title(mytitle)
    plt.xlabel("Bucket")
    plt.ylabel("Value")
    plt.tick_params(labelleft=True) #to anonymize set to False
    #plt.ylim(bottom=-6000000,top=2000000) #to disable autoscale
    plt.legend()
    plt.show()

    zr_col = chflibor_all_ipl.columns[pos]

    #if dtc > 2:
    if dtc > 1:
        plt.figure()
        plt.scatter(chflibor_all_ipl.index,chflibor_all_ipl[zr_col],label=zr_col,s=2,color='royalblue')
        plt.scatter(chflibor_all_ipl.index,chflibor_all_ipl[prev_zr_col],label=prev_zr_col,s=2,color='lightgrey')
        plt.scatter(chflibor_all_ipl_deltas.index,chflibor_all_ipl_deltas[zr_col],label="Delta",s=2,color='green')
        plt.title("CHF Libor Interpolated")
        plt.xlabel("Days Forward")
        plt.ylabel("Zero Rate")
        plt.tick_params(labelleft=True) #to anonymize set to False
        plt.legend()
        plt.show()
    
    if dtc > 1:
        plt.figure(figsize=(12,7))
        plt.bar(chflibor_all_ipl_bucket_avg.index,chflibor_all_ipl_bucket_avg[prev_zr_col]
               ,label=prev_zr_col,width=0.3,color='lightgrey',align='center')
        plt.bar(chflibor_all_ipl_bucket_avg.index,chflibor_all_ipl_bucket_avg[zr_col]
               ,label=zr_col,width=-0.3,color='royalblue',align='edge')
        plt.bar(chflibor_all_ipl_deltas_bucket_avg.index,chflibor_all_ipl_deltas_bucket_avg[zr_col]
               ,label="Delta",width=0.3,color='green',align='edge')
        plt.title("CHF Libor Interpolated Average per Bucket")
        plt.xlabel("Bucket")
        plt.ylabel("Zero Rate")
        plt.tick_params(labelleft=True) #to anonymize set to False
        plt.legend()
        plt.show()
    
    if dtc == 1:
        chf_swaps_flattened_ts_bucket_sum_all = chf_swaps_flattened_ts_bucket_sum.copy()
    else:
        chf_swaps_flattened_ts_bucket_sum_all = pd.concat([chf_swaps_flattened_ts_bucket_sum_all,chf_swaps_flattened_ts_bucket_sum],axis=1)

    prev_dt_col_bs = dt_col_bs
    prev_zr_col    = zr_col

### Test Violin Plot

vio1 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='1M']
vio1 = vio1.reset_index(drop=True)
vio2 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='1Y']
vio2 = vio2.reset_index(drop=True)
vio3 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='3Y']
vio3 = vio3.reset_index(drop=True)
vio4 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='5Y']
vio4 = vio4.reset_index(drop=True)
vio5 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='10Y']
vio5 = vio5.reset_index(drop=True)
vio6 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='30Y']
vio6 = vio6.reset_index(drop=True)
vio7 = chflibor_all_ipl_bucket['ZERO_RATE_2019-12-31'].loc[chflibor_all_ipl_bucket['BUCKET_LABEL']=='50Y']
vio7 = vio7.reset_index(drop=True)

plt.figure()
plt.violinplot([vio1,vio2,vio3,vio4,vio5,vio6,vio7])
plt.show()

