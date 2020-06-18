# Smart Hedging Main Routine

import os
import pandas as pd

wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile_assets = "/valuations_assets_all.xlsx"
testfile_swaps = "/valuations_swaps_all.xlsx"
full_path_assets = datadir+testfile_assets
full_path_swaps = datadir+testfile_swaps

all_data_needed_assets_upto_2011 = pd.read_excel(full_path_assets
                                       ,sheet_name = 'VAL_ASSETS_ALL_UPTO_2011'
                                       ,usecols=['VAL_DATE','SUM(PV.CLEAN_PV)'])

all_data_needed_assets_since_2012 = pd.read_excel(full_path_assets
                                       ,sheet_name = 'VAL_ASSETS_ALL_SINCE_2012'
                                       ,usecols=['VAL_DATE','SUM(PV.CLEAN_PV)'])

all_data_needed_swaps = pd.read_excel(full_path_swaps
                                       ,sheet_name = 'VALUATIONS_SWAPS_ALL'
                                       ,usecols=['VAL_DATE','SUM(PV.CLEAN_PV)'])

all_data_needed_assets = all_data_needed_assets_upto_2011.append(all_data_needed_assets_since_2012
                                                                ,ignore_index=True)

all_data_needed_assets.columns = ['SUM_ASSETS','VAL_DATE']
all_data_needed_swaps.columns = ['SUM_SWAPS','VAL_DATE']

result_table_assets = pd.DataFrame(all_data_needed_assets.groupby('VAL_DATE').sum())
result_table_swaps = pd.DataFrame(all_data_needed_swaps.groupby('VAL_DATE').sum())

result_table_assets.plot()
result_table_swaps.plot()

ax = result_table_assets.plot()
result_table_swaps.plot(ax=ax)
