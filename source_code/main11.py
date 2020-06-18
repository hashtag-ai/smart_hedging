# Smart Hedging Main Routine

import os
import pandas as pd
import matplotlib.pyplot as plt

wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

file_chflibor = "/chflibor_monthly.xlsx"
full_path_chflibor = datadir+file_chflibor

chflibor = pd.read_excel(full_path_chflibor
                        ,sheet_name = 'CHFLIBOR'
                        ,usecols=['VAL_DATE','DAYS_FWD','ZERO_RATE'])

#chflibor_plot = pd.DataFrame(chflibor[(chflibor['VAL_DATE'].dt.date >= pd.to_datetime("2000-12-31"))
#                                    & (chflibor['VAL_DATE'].dt.month == 12)])

chflibor_plot = pd.DataFrame(chflibor[(chflibor['VAL_DATE'].dt.date >= pd.to_datetime("2019-01-01"))])

vd = chflibor_plot.groupby(['VAL_DATE'])

plt.figure(1)
for i, data in vd:
    chflibor_subplot = pd.DataFrame(chflibor_plot[(chflibor_plot['VAL_DATE'].dt.date == pd.to_datetime(i))])
    plt.plot(chflibor_subplot['DAYS_FWD']
            ,chflibor_subplot['ZERO_RATE']
            ,label=pd.to_datetime(i).date())

plt.title("CHF Libor")
plt.xlabel("Days Forward")
plt.ylabel("Zero Rate")
plt.legend()
plt.show()

plt.figure(num=2,figsize=(40,4))
ax0=plt.subplot(1,vd.ngroups,1)
ind=0
for i, data in vd:
    ind=ind+1
    chflibor_subplot = pd.DataFrame(chflibor_plot[(chflibor_plot['VAL_DATE'].dt.date == pd.to_datetime(i))])
    plt.subplot(1,vd.ngroups,ind,sharex=ax0,sharey=ax0)
    plt.plot(chflibor_subplot['DAYS_FWD'],chflibor_subplot['ZERO_RATE'])
    plt.title(pd.to_datetime(i).date())
    plt.xlabel("Days Forward")
    if ind==1: plt.ylabel("Zero Rate")
    #else: plt.yticks([])
plt.suptitle("CHF Libor")
plt.show()