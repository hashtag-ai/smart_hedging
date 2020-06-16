# Smart Hedging Main Routine

import os
import pandas as pd
import matplotlib.pyplot as plt

wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile_assets = "/valuations_assets_active_minus_principal.xlsx"
testfile_swaps = "/valuations_swaps_active.xlsx"
full_path_assets = datadir+testfile_assets
full_path_swaps = datadir+testfile_swaps

all_data_needed_assets = pd.read_excel(full_path_assets
                                       ,sheet_name = 'VAL_ASSETS_ACTIVE_MINUS_PRINCIP'
                                       ,usecols=['VAL_DATE','SUM(PV.CLEAN_PV)-AVG(PCF_AMT)'])

all_data_needed_swaps = pd.read_excel(full_path_swaps
                                       ,sheet_name = 'VALUATIONS_SWAPS_ACTIVE'
                                       ,usecols=['VAL_DATE','SUM(PV.CLEAN_PV)'])

all_data_needed_assets.columns = ['SUM_ASSETS','VAL_DATE']
all_data_needed_swaps.columns = ['SUM_SWAPS','VAL_DATE']

result_table_assets = pd.DataFrame(all_data_needed_assets.groupby('VAL_DATE').sum())
result_table_swaps_precise = pd.DataFrame(all_data_needed_swaps.groupby('VAL_DATE').sum())
result_table_swaps = pd.DataFrame(all_data_needed_swaps
                                  [all_data_needed_swaps['VAL_DATE'].dt.month == 12]
                                  .groupby('VAL_DATE').sum())

plt.plot(result_table_swaps_precise.index,result_table_swaps_precise['SUM_SWAPS'],label="Swaps detailed")
plt.plot(result_table_assets.index,result_table_assets['SUM_ASSETS'],label="Assets yearly")
plt.plot(result_table_swaps.index,result_table_swaps['SUM_SWAPS'],label="Swaps yearly")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
