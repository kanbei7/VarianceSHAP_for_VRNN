import os
import sys
import json
import pandas as pd
import numpy as np
import pickle as pk
import datetime as dt
import discretization_utils as utils

from tqdm import tqdm


str2dts = dt.datetime.strptime
date_fmt = '%Y-%m-%d'
dts_fmt = '%Y-%m-%d %H:%M:%S'

with open(os.path.join('outputs','chart-store_dict.json'),'r') as f:
	CHART_STORE_dict = json.load(f)

with open(os.path.join('outputs','value-num_dict.json'),'r') as f:
	VALUE_NUM_dict = json.load(f)


range_df = pd.read_csv(os.path.join('outputs','range_limit.csv'))
RANGE_dict = {}
for a,b in zip(range_df['feat'],range_df['max']):
	RANGE_dict[a] = float(b)

def isFloat(x):
	try:
		f = float(x)
	except:
		return False
	return True


features = []

for sid in tqdm(os.listdir('stays_intermediate')):
	lab_df = pd.read_csv(os.path.join('stays_intermediate',sid,'labs.csv'))
	if len(lab_df) == 0:
		continue
	
	chart_df = pd.read_csv(os.path.join('stays_intermediate',sid,'charts.csv'))
	assert(len(chart_df)>0)
	lab_df = lab_df[['newname', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom']]
	chart_df = chart_df[['newname', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom']]
	combined_df = pd.concat([lab_df, chart_df])
	combined_df = combined_df[(combined_df['value'].notnull()) | (combined_df['valuenum'].notnull())]
	
	res_df = []
	for item_name in combined_df['newname'].unique():
		if not item_name in CHART_STORE_dict.keys():
			continue
		tmp_df = combined_df[(combined_df['newname'] == item_name) & (combined_df[VALUE_NUM_dict[item_name]].notnull())].copy()

		if len(tmp_df)==0:
			print(item_name)
			print(VALUE_NUM_dict[item_name])
			print(len(combined_df[(combined_df['newname'] == item_name) & (combined_df['value'].notnull())]))
			print(len(combined_df[(combined_df['newname'] == item_name) & (combined_df['valuenum'].notnull())]))
			tmp_df = combined_df[combined_df['newname'] == item_name].copy()
			tmp_df[VALUE_NUM_dict[item_name]] = [a if pd.isna(b) else b for a, b in zip(tmp_df.value, tmp_df.valuenum)]			
			print(list(tmp_df[VALUE_NUM_dict[item_name]].unique())[:5])
			continue
		assert(len(tmp_df)>0)
		#convert units
		#temp F to C
		if item_name=='Temp':
			tmp_df['valuenum'] = [utils.Temperature_F2C(float(a)) if (not pd.isna(a)) and (not pd.isna(b)) and (float(a)>60 or 'f' in b.lower()) else a for a,b in zip(tmp_df['valuenum'], tmp_df['valueuom'])]

		#chloride-blood mg/dL to mEq/L
		if item_name=='Chloride-Blood':
			tmp_df['valuenum'] = [utils.Chloride_mgdl2mEqL(float(a)) if (not pd.isna(a)) and (not pd.isna(b)) and ('mg' in b.lower()) else a for a,b in zip(tmp_df['valuenum'], tmp_df['valueuom'])]


		tmp_df = tmp_df[['newname', CHART_STORE_dict[item_name], VALUE_NUM_dict[item_name]]]
		tmp_df.columns = ['newname','time','value']
		res_df.append(tmp_df.copy())
	
	#to ['newname','time','value']
	res_df = pd.concat(res_df)
	#all name to lower, ph-arterial
	res_df['newname'] = res_df['newname'].apply(lambda x: x.strip().lower())
	res_df['newname'] = res_df['newname'].apply(lambda x: 'ph-blood' if x =='ph-arterial' else x)
	res_df['newname'] = res_df['newname'].apply(lambda x: 'gcs-motor' if x =='gcs_motor' else x)
	res_df['newname'] = res_df['newname'].apply(lambda x: 'inr' if x =='inr-blood' else x)
	res_df.drop_duplicates(inplace = True, ignore_index = True)

	new_values = []
	for k,v in zip(res_df['newname'],res_df['value']):
		# abpdia, abpmean, abpsys, nbpdia, nbpmean, 10000<bp<300000, bp/=1000
		if k in ['abpdia', 'abpmean', 'abpsys', 'nbpdia', 'nbpmean']:
			new_values.append(v/1000 if float(v) >10000 and float(v)<300000 else v)
			continue
		# fiO2<1 then*100
		if k == 'fio2':
			new_values.append(v*100 if v<1 else v)
			continue
	
		# 100<spo2<=10000, spo2 /=100
		if k == 'spo2':
			new_values.append(v/100 if v<=10000 and v>100 else v)
			continue

		# hematocrit-blood>100, /=10
		if 'hematocrit' in k:
			new_values.append(v/10 if v>100 else v)
			continue

		# drop base excess-blood
		if k == 'base excess-blood':
			new_values.append(np.nan)
			continue

		# drop all >99999 or negatives
		if isFloat(v):
			new_values.append(np.nan if float(v)>99999 or float(v)<0 else v)
			continue

		new_values.append(v)

	res_df['value'] = new_values

	#impose range limit
	new_values = []
	for k,v in zip(res_df['newname'],res_df['value']):
		if k in RANGE_dict.keys():
			new_values.append(np.nan if float(v)>RANGE_dict[k] or float(v)<0 else v)
			continue
		new_values.append(v)
	res_df['value'] = new_values

	res_df = res_df[res_df['value'].notnull()]
	if len(res_df)==0:
		continue
	assert(len(res_df)>0)
	features.extend([feat for feat in list(res_df['newname'].unique()) if not feat in features])

	#write
	info_df = pd.read_csv(os.path.join('stays_intermediate',sid,'info.csv'))
	vent_df = pd.read_csv(os.path.join('stays_intermediate',sid,'vent.csv'))
	if  not os.path.exists(os.path.join('stays_final', sid)):
		os.mkdir(os.path.join('stays_final', sid))


	res_df.to_csv(os.path.join('stays_final',sid,'timeseries.csv'), index = False)
	info_df.to_csv(os.path.join('stays_final',sid,'info.csv'), index = False)
	vent_df.to_csv(os.path.join('stays_final',sid,'vent.csv'), index = False)

print(len(features))
print(features)
