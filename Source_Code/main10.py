# Smart Hedging Main Routine

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

wdir = os.getcwd()
datadir = os.path.join(wdir,"data\TPC")

fas133_data = "/fas133_data_*.xlsx"
full_path_fas133_data = datadir+fas133_data

all_fas_files = glob.glob(full_path_fas133_data)

all_data_fas133 = []

for filename in all_fas_files:
    all_data_fas133_temp = pd.read_excel(filename
                                        ,sheet_name = 'Sheet1'
                                        ,usecols=['COB Date','Deal No','CCY','L/S','RATE_TYPE'
                                                 ,'CLEAN_NPV','EVENT_02','EVENT_11'
                                                 ,'EVENT_03','EVENT_12'
                                                 ])
    all_data_fas133.append(all_data_fas133_temp)

all_data_fas133 = pd.concat(all_data_fas133)

pay_swaps = all_data_fas133['Deal No'].loc[(all_data_fas133['L/S'] == "S") &
                                  (all_data_fas133['RATE_TYPE'] == "FIXED")]
rec_swaps = all_data_fas133['Deal No'].loc[(all_data_fas133['L/S'] == "S") &
                                  (all_data_fas133['RATE_TYPE'] == "FLOAT")]
pay_swaps.drop_duplicates(inplace=True)
rec_swaps.drop_duplicates(inplace=True)

all_pay_swaps_fas133 = all_data_fas133.loc[all_data_fas133['Deal No'].isin(pay_swaps)]
all_rec_swaps_fas133 = all_data_fas133.loc[all_data_fas133['Deal No'].isin(rec_swaps)]

all_pay_swaps_fas133 = all_pay_swaps_fas133.drop(columns=['Deal No','RATE_TYPE'],axis=1)
all_rec_swaps_fas133 = all_rec_swaps_fas133.drop(columns=['Deal No','RATE_TYPE'],axis=1)
all_data_fas133 = all_data_fas133.drop(columns=['Deal No','RATE_TYPE'],axis=1)

result_table_pay = pd.DataFrame(all_pay_swaps_fas133
                                  [all_pay_swaps_fas133['CCY'] == "CHF"]
                                  .groupby('COB Date').sum())
result_table_rec = pd.DataFrame(all_rec_swaps_fas133
                                  [all_rec_swaps_fas133['CCY'] == "CHF"]
                                  .groupby('COB Date').sum())
result_table_all = pd.DataFrame(all_data_fas133
                                  [all_data_fas133['CCY'] == "CHF"]
                                  .groupby('COB Date').sum())

result_table_pay.sort_index(ascending=True,inplace=True)
result_table_rec.sort_index(ascending=True,inplace=True)
result_table_all.sort_index(ascending=True,inplace=True)

result_table_pay_deltas = result_table_pay.diff()
result_table_rec_deltas = result_table_rec.diff()
result_table_all_deltas = result_table_all.diff()

result_table_pay_pct_deltas = result_table_pay.pct_change()
result_table_rec_pct_deltas = result_table_rec.pct_change()
result_table_all_pct_deltas = result_table_all.pct_change()

plt.figure(1)
plt.plot(result_table_pay.index,result_table_pay['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_pay.index,result_table_pay['EVENT_02'],label="Event 02")
plt.plot(result_table_pay.index,result_table_pay['EVENT_11'],label="Event 11")

plt.title("Payer Swaps Only")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(result_table_rec.index,result_table_rec['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_rec.index,result_table_rec['EVENT_02'],label="Event 02")
plt.plot(result_table_rec.index,result_table_rec['EVENT_11'],label="Event 11")

plt.title("Receiver Swaps Only")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(3)
plt.plot(result_table_all.index,result_table_all['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_all.index,result_table_all['EVENT_02'],label="Event 02")
plt.plot(result_table_all.index,result_table_all['EVENT_11'],label="Event 11")
#plt.plot(result_table_all.index,result_table_all['EVENT_03'],label="Event 03")
#plt.plot(result_table_all.index,result_table_all['EVENT_12'],label="Event 12")

plt.title("All Swaps")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

#plt.figure(4)
#plt.stackplot(result_table.index,result_table['CLEAN_NPV']
#                                 ,result_table['EVENT_02']
#                                 ,result_table['EVENT_11']
#                                 ,labels=["Clean NPV","Event 02","Event 11"])
#
#plt.xlabel("Date")
#plt.ylabel("Value")
#plt.legend()
#plt.show()

plt.figure(5)
plt.plot(result_table_pay_deltas.index,result_table_pay_deltas['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_pay_deltas.index,result_table_pay_deltas['EVENT_02'],label="Event 02")
#plt.plot(result_table_pay_deltas.index,result_table_pay_deltas['EVENT_11'],label="Event 11")

plt.title("Payer Swaps Only - Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(6)
plt.plot(result_table_rec_deltas.index,result_table_rec_deltas['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_rec_deltas.index,result_table_rec_deltas['EVENT_02'],label="Event 02")
#plt.plot(result_table_rec_deltas.index,result_table_rec_deltas['EVENT_11'],label="Event 11")

plt.title("Receiver Swaps Only - Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(7)
plt.plot(result_table_all_deltas.index,result_table_all_deltas['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_all_deltas.index,result_table_all_deltas['EVENT_02'],label="Event 02")
#plt.plot(result_table_all_deltas.index,result_table_all_deltas['EVENT_11'],label="Event 11")

plt.title("All Swaps - Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
#plt.ticklabel_format(style='plain',axis='y') #anonymize
#plt.tick_params(labelleft=False)             #anonymize
plt.legend()
plt.show()

plt.figure(8)
plt.plot(result_table_pay_pct_deltas.index,result_table_pay_pct_deltas['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_pay_pct_deltas.index,result_table_pay_pct_deltas['EVENT_02'],label="Event 02")
#plt.plot(result_table_pay_pct_deltas.index,result_table_pay_pct_deltas['EVENT_11'],label="Event 11")

plt.title("Payer Swaps Only - Percentage Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(9)
plt.plot(result_table_rec_pct_deltas.index,result_table_rec_pct_deltas['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_rec_pct_deltas.index,result_table_rec_pct_deltas['EVENT_02'],label="Event 02")
#plt.plot(result_table_rec_pct_deltas.index,result_table_rec_pct_deltas['EVENT_11'],label="Event 11")

plt.title("Receiver Swaps Only - Percentage Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(10)
plt.plot(result_table_all_pct_deltas.index,result_table_all_pct_deltas['CLEAN_NPV'],label="Clean NPV")
plt.plot(result_table_all_pct_deltas.index,result_table_all_pct_deltas['EVENT_02'],label="Event 02")
#plt.plot(result_table_all_pct_deltas.index,result_table_all_pct_deltas['EVENT_11'],label="Event 11")

plt.title("All Swaps - Percentage Deltas")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()