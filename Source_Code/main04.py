# Smart Hedging Main Routine

import os
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.pyplot as plt
import pandas as pd
wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

testfile = "\chflibor_smaller_sample_annual_year_only.xlsx"
all = datadir+testfile

all_data = pd.read_excel(all)
print(all_data)

ax1 = Axes3D(figure())
ax2 = Axes3D(figure())
x = all_data['DAYS_FWD']
y = all_data['VAL_DATE_YEAR']
z = all_data['ZERO_RATE']
ax1.plot(x, y, z)
ax2.plot(x, y, z, '.')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
show()