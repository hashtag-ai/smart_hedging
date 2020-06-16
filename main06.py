# Smart Hedging Main Routine

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile = "\chflibor_smaller_sample_annual_year_only.xlsx"
testfile2 = "\chflibor_smaller_sample_annual_year_only_days_fwd_only.xlsx"
testfile3 = "\chflibor_smaller_sample_annual_year_only_years_only.xlsx"
all = datadir+testfile
all2 = datadir+testfile2
all3 = datadir+testfile3

all_data = pd.read_excel(all)
all_data2 = pd.read_excel(all2)
all_data3 = pd.read_excel(all3)
print(all_data)
print(all_data2)
print(all_data3)

fig = plt.figure()
ax = plt.axes(projection="3d")

x = all_data2['DAYS_FWD_SINGLE']
y = all_data3['YEARS']
z = all_data['ZERO_RATE']

X, Y = np.meshgrid(x, y)
Z = np.array([z]).reshape(4,41)

ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('Days Forward')
ax.set_ylabel('Year')
ax.set_zlabel('Zero Rate')

plt.show()