# Smart Hedging Main Routine

import os
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile = "\chflibor_sample.xlsx"
all = datadir+testfile

# =============================================================================
# with open(all,'r') as f:
#     for line in f:
#         print(line)
#     f.close()
# =============================================================================

# =============================================================================
# with open(all,'r') as f:
#     all_data = f.readlines()
#     f.close()
# 
# print(all_data)
# =============================================================================

# =============================================================================
# wb = xlrd.open_workbook(all)
# sheet = wb.sheet_by_name("CHFLIBOR")
# print(sheet.nrows, sheet.ncols)
# =============================================================================

all_data = pd.read_excel(all)
print(all_data)
plt.scatter(all_data['DAYS_FWD'],all_data['ZERO_RATE'])
plt.title("CHF Libor Rate")
plt.xlabel("Days Forward")
plt.ylabel("Interest Rate")
plt.show()