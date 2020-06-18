# Smart Hedging Main Routine

import os
import matplotlib.pyplot as plt
import pandas as pd
wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile = "\chflibor_sample_annual.xlsx"
all = datadir+testfile

all_data = pd.read_excel(all)
print(all_data)
plt.scatter(all_data['DAYS_FWD'],all_data['ZERO_RATE'])
plt.title("CHF Libor Rate")
plt.xlabel("Days Forward")
plt.ylabel("Interest Rate")
plt.show