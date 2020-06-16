# Smart Hedging Main Routine

import os
import pandas as pd

wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile_active = "\pnl_assets_active.xlsx"
testfile_historic = "\pnl_assets_historic.xlsx"
full_path_active = datadir+testfile_active
full_path_historic = datadir+testfile_historic

all_data_needed_active = pd.read_excel(full_path_active
                                       ,sheet_name = 'PNL_ASSETS_ACTIVE'
                                       ,usecols=['VAL_DATE','SUM(AC.AMT)'])
all_data_needed_historic = pd.read_excel(full_path_historic
                                         ,sheet_name = 'PNL_ASSETS_HISTORIC'
                                         ,usecols=['VAL_DATE','SUM(AC.AMT)'])

all_data_needed_active.columns = ['SUM_ACTIVE','VAL_DATE']
all_data_needed_historic.columns = ['SUM_HISTORIC','VAL_DATE']

result_table_active = pd.DataFrame(all_data_needed_active.groupby('VAL_DATE').sum())
result_table_historic = pd.DataFrame(all_data_needed_historic.groupby('VAL_DATE').sum())
result_table = pd.concat([result_table_active,result_table_historic],axis=1)
result_table_new = result_table.copy()
result_table_new['SUM_BOTH'] = result_table_new['SUM_ACTIVE'] + result_table_new['SUM_HISTORIC']

#result_table_active.plot()
#result_table_historic.plot()
result_table_new.plot()
result_table_new.plot.area(y=['SUM_ACTIVE','SUM_BOTH'],stacked=False)
