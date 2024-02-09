import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
import datetime as dt
from tqdm import tqdm

str2dts = dt.datetime.strptime
date_fmt = '%Y-%m-%d'
dts_fmt = '%Y-%m-%d %H:%M:%S'

with open(os.path.join('outputs','labitems_names.pkl'),'rb') as f:
	LAB_ITEMS_dict = pk.load(f)
LAB_ITEMS_SET = list(LAB_ITEMS_dict.keys())

with open(os.path.join('outputs','chartitems_names.pkl'),'rb') as f:
	CHART_ITEMS_dict = pk.load(f)
CHART_ITEMS_SET = list(CHART_ITEMS_dict.keys())

#ITEM_NAMES = [LAB_ITEMS_dict[k] for k in LAB_ITEMS_SET] + [CHART_ITEMS_dict[k] for k in CHART_ITEMS_SET]
ITEM_NAMES = ['WBC-Blood',
		'Urine Color',
		'Urea Nitrogen-Blood',
		'Temp Site',
		'Temp',
		'SpO2',
		'Sodium-serum',
		'Sodium-Blood',
		'RR',
		'RBC-Blood',
		'PTT',
		'PT',
		'Potassium-serum',
		'Potassium-Blood',
		'pO2-Blood',
		'Platelet abs-Blood',
		'Phosphate-Blood',
		'PH-venous',
		'pH-Blood',
		'PH-arterial',
		'pCO2-Blood',
		'NBPmean',
		'NBPdia',
		'Lactate-Blood',
		'INR-Blood',
		'INR',
		'HR',
		'Hemoglobin-Blood',
		'Hematocrit-serum',
		'Hematocrit-Blood',
		'HCO3-serum',
		'Glucose-serum',
		'Glucose-Blood',
		'GCS-Verbal',
		'GCS-Eye',
		'GCS_Motor',
		'FiO2',
		'Creatinine-serum',
		'Creatinine-Blood',
		'CO2 Total-Blood',
		'Chloride-serum',
		'Chloride-Blood',
		'Calcium Total-Blood',
		'Calcium Free-Blood',
		'Braden-Sense',
		'Braden-Nut',
		'Braden-Moist',
		'Braden-Mob',
		'Braden-Fric',
		'Braden-Act',
		'Base Excess-Blood',
		'AST-Blood',
		'ALT-Blood',
		'ABPsys',
		'ABPmean',
		'ABPdia'
		]
ITEM_NAMES = set(ITEM_NAMES)

LABS_DF = []
CHARTS_DF = []

for sid in tqdm(os.listdir('stays_intermediate')):
	lab_df = pd.read_csv(os.path.join('stays_intermediate',sid,'labs.csv'))
	if len(lab_df)>0:
		lab_df['stay_id'] = [int(sid)]*len(lab_df)
		lab_df = lab_df[['stay_id','newname','charttime','storetime']]
		LABS_DF.append(lab_df.copy())

	chart_df = pd.read_csv(os.path.join('stays_intermediate',sid,'charts.csv'))
	if len(chart_df)>0:
		chart_df = chart_df[['stay_id','newname','charttime','storetime']]
		CHARTS_DF.append(chart_df.copy())


LABS_DF = pd.concat(LABS_DF)
CHARTS_DF = pd.concat(CHARTS_DF)
print(LABS_DF.dtypes)
print(CHARTS_DF.dtypes)

def calc_timediff_hrs(t1,t2):
	t1 = str2dts(t1, dts_fmt)
	t2 = str2dts(t2, dts_fmt)
	return (t1-t2).total_seconds()/3600


def check_timediff(id_key, col_name):
	global LABS_DF
	global CHARTS_DF
	timediff = []
	lab_tmp = LABS_DF[(LABS_DF[col_name] == id_key) & (LABS_DF['charttime'].notnull()) & (LABS_DF['storetime'].notnull()) ]
	chart_tmp = CHARTS_DF[(CHARTS_DF[col_name] == id_key) & (CHARTS_DF['charttime'].notnull()) & (CHARTS_DF['storetime'].notnull()) ]

	if len(lab_tmp)==0 and len(chart_tmp)==0:
		return -1.0, -1.0, -1.0, -1.0

	if len(lab_tmp)>0:
		timediff.extend([calc_timediff_hrs(a,b) for a,b  in zip(lab_tmp['storetime'] , lab_tmp['charttime'])])

	if len(chart_tmp)>0:
		timediff.extend([calc_timediff_hrs(a,b) for a,b  in zip(chart_tmp['storetime'] , chart_tmp['charttime'])])

	return np.mean(timediff), max(timediff), min(timediff), np.median(timediff)

#per name level
NAMES_MEAN = {}
NAMES_MAX = {}
NAMES_MIN = {}
NAMES_MEDIAN = {}

for item_name in tqdm(ITEM_NAMES):
	#print('Start: ' + item_name)
	NAMES_MEAN[item_name], NAMES_MAX[item_name], NAMES_MIN[item_name], NAMES_MEDIAN[item_name] = check_timediff(item_name, 'newname')
	#print('\n')


with open(os.path.join('outputs', 'Chart-Store_timediff_by_names_subset.csv'), 'w') as f:
	f.writelines([','.join([k, '%.4f'%NAMES_MEAN[k], '%.4f'%NAMES_MAX[k], '%.4f'%NAMES_MIN[k], '%.4f'%NAMES_MEDIAN[k]  ])  +'\n' for k in NAMES_MEAN.keys()])


