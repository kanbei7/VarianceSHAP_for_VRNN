import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
import datetime as dt
from tqdm import tqdm

with open(os.path.join('outputs','labitems_names.pkl'),'rb') as f:
	LAB_ITEMS_dict = pk.load(f)
LAB_ITEMS_SET = set(LAB_ITEMS_dict.keys())

with open(os.path.join('outputs','chartitems_names.pkl'),'rb') as f:
	CHART_ITEMS_dict = pk.load(f)
CHART_ITEMS_SET = set(CHART_ITEMS_dict.keys())

VENTILATION_df = pd.read_csv(os.path.join('outputs','ventilation.csv'))

'''
perstay_labs/
-hadmilabs_[hadmid].csv


stays_raw/
-stayid/
--info.csv
--stay[stayid].csv

'''


HADMID_SET = set([int(x.split('.')[0].split('_')[-1]) for x in os.listdir('perstay_labs')])
print(len(HADMID_SET))
abnormal_cnt = 0
zero_labs_cnt = 0
zero_charts_cnt = 0
for sid in tqdm(os.listdir('stays_raw')):
	stay_info_df = pd.read_csv(os.path.join('stays_raw',sid,'info.csv'))
	assert(len(stay_info_df)==1)
	hid = list(stay_info_df['hadm_id'])[0]
	if not hid in HADMID_SET:
		continue

	#deal labs
	intime = list(stay_info_df['intime'])[0]
	outtime = list(stay_info_df['outtime'])[0]
	if outtime<intime:
		print(sid)
		print(intime)
		print(outtime)
		abnormal_cnt+=1
		continue
	labs_df = pd.read_csv(os.path.join('perstay_labs', 'hadmlabs_%d.0.csv'%hid))
	labs_df = labs_df[labs_df.itemid.isin(LAB_ITEMS_SET)]
	labs_df = labs_df[((labs_df['charttime']>=intime) & (labs_df['charttime']<=outtime)) | (labs_df['storetime']<=outtime)]

	if len(labs_df)==0:
		zero_labs_cnt +=1
	else:
		labs_df['newname'] = labs_df['itemid'].apply(lambda x:LAB_ITEMS_dict[x])

	pth = os.path.join('stays_intermediate', sid)
	if not os.path.exists(pth):
		os.mkdir(pth)
	labs_df.to_csv(os.path.join(pth, 'labs.csv'), index = False)
	stay_info_df.to_csv(os.path.join(pth, 'info.csv'), index = False)

	#deal chart events
	chart_df = pd.read_csv(os.path.join('stays_raw',sid,'stay'+sid+'.csv'))
	chart_df = chart_df[chart_df['itemid'].isin(CHART_ITEMS_SET)]

	if len(chart_df)==0:
		zero_charts_cnt +=1
	else:
		chart_df['newname'] = chart_df['itemid'].apply(lambda x:CHART_ITEMS_dict[x])

	chart_df.to_csv(os.path.join(pth, 'charts.csv'), index = False)
	
	#deal vent
	vent_df = VENTILATION_df[VENTILATION_df.stay_id == int(sid)]
	vent_df.to_csv(os.path.join(pth, 'vent.csv'), index = False)

print('#stays intime > outtime: %d'%abnormal_cnt)
print('#stays w.o. labs: %d'%zero_labs_cnt)
print('#stays w.o. charts: %d'%zero_charts_cnt)