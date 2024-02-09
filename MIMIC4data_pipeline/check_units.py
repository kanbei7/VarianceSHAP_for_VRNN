import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
import datetime as dt
from tqdm import tqdm

with open(os.path.join('outputs','labitems_names.pkl'),'rb') as f:
	LAB_ITEMS_dict = pk.load(f)
LAB_ITEMS_SET = list(LAB_ITEMS_dict.keys())

with open(os.path.join('outputs','chartitems_names.pkl'),'rb') as f:
	CHART_ITEMS_dict = pk.load(f)
CHART_ITEMS_SET = list(CHART_ITEMS_dict.keys())

ITEM_NAMES = [LAB_ITEMS_dict[k] for k in LAB_ITEMS_SET] + [CHART_ITEMS_dict[k] for k in CHART_ITEMS_SET]
ITEM_NAMES = set(ITEM_NAMES)

LABS_DF = []
CHARTS_DF = []
PA_LOS = {}
for sid in tqdm(os.listdir('stays_intermediate')):
	lab_df = pd.read_csv(os.path.join('stays_intermediate',sid,'labs.csv'))
	if len(lab_df)>0:
		lab_df['stay_id'] = [int(sid)]*len(lab_df)
		lab_df = lab_df[['subject_id','stay_id','itemid','newname','value','valuenum','valueuom']]
		LABS_DF.append(lab_df.copy())

	chart_df = pd.read_csv(os.path.join('stays_intermediate',sid,'charts.csv'))
	if len(chart_df)>0:
		chart_df = chart_df[['subject_id','stay_id','itemid','value','valuenum','valueuom','newname']]
		CHARTS_DF.append(chart_df.copy())

	info = pd.read_csv(os.path.join('stays_intermediate',sid,'info.csv'))
	PA_LOS[int(sid)] = list(info['LOS_hrs'])[0]

LABS_DF = pd.concat(LABS_DF)
CHARTS_DF = pd.concat(CHARTS_DF)
print(LABS_DF.dtypes)
print(CHARTS_DF.dtypes)


def check_unit(id_key, col_name):
	global LABS_DF
	global CHARTS_DF
	units = []
	lab_tmp = LABS_DF[(LABS_DF[col_name] == id_key) & (LABS_DF['valueuom'].notnull())]
	chart_tmp = CHARTS_DF[(CHARTS_DF[col_name] == id_key) & (CHARTS_DF['valueuom'].notnull())]
	if len(lab_tmp)>0 and len(lab_tmp['valueuom'].unique())>0:
		units.extend(list(lab_tmp['valueuom'].unique()))

	if len(chart_tmp)>0 and len(chart_tmp['valueuom'].unique())>0:
		units.extend(list(chart_tmp['valueuom'].unique()))

	palst = []
	stayslst = []
	lab_tmp = LABS_DF[(LABS_DF[col_name] == id_key) & ((LABS_DF['valuenum'].notnull()) | (LABS_DF['value'].notnull()) )]
	chart_tmp = CHARTS_DF[(CHARTS_DF[col_name] == id_key) & ((CHARTS_DF['valuenum'].notnull()) | (CHARTS_DF['value'].notnull()) )]
	if len(lab_tmp)>0:
		palst.extend(list(lab_tmp['subject_id'].unique()))
		stayslst.extend(list(lab_tmp['stay_id'].unique()))
	if len(chart_tmp)>0:
		palst.extend(list(chart_tmp['subject_id'].unique()))
		stayslst.extend(list(chart_tmp['stay_id'].unique()))
	if len(lab_tmp) + len(chart_tmp) ==0:
		return [], 0, 0, -1

	n_pa = len(set(palst))
	n_stays = len(set(stayslst))
	avg_m_perhr = (len(lab_tmp) + len(chart_tmp))/sum([PA_LOS[sid] for sid in set(stayslst)])
	#all unique units, #pa, # stays, avg #m per hr
	return list(set(units)), n_pa, n_stays, avg_m_perhr

#per name level
NAME_UNITS_DICT = {}
NAMES_N_PA = {}
NAMES_N_STAY = {}
NAMES_FREQ = {}

for item_name in tqdm(ITEM_NAMES):
	#print('Start: ' + item_name)
	NAME_UNITS_DICT[item_name], NAMES_N_PA[item_name], NAMES_N_STAY[item_name], NAMES_FREQ[item_name] = check_unit(item_name, 'newname')
	#print('\n')

print('\n =============== \n')

#per item level
UNITS_DICT = {}
ITEMID_N_PA = {}
ITEMID_N_STAY = {}
ITEMID_FREQ = {}
for item_id in tqdm(LAB_ITEMS_SET + CHART_ITEMS_SET):
	#print('Start: %d'%item_id)
	UNITS_DICT[item_id], ITEMID_N_PA[item_id], ITEMID_N_STAY[item_id], ITEMID_FREQ[item_id] = check_unit(item_id, 'itemid')
	#print('\n')


with open(os.path.join('outputs', 'unitsNavail_by_names.csv'), 'w') as f:
	f.writelines([ ','.join([str(k), str(NAMES_N_PA[k]), str(NAMES_N_STAY[k]), '%.8f'%NAMES_FREQ[k],'|'.join(NAME_UNITS_DICT[k])]) +'\n' for k in NAME_UNITS_DICT.keys()])


with open(os.path.join('outputs', 'unitsNavail_by_itemid.csv'), 'w') as f:
	f.writelines([ ','.join([str(k), str(ITEMID_N_PA[k]), str(ITEMID_N_STAY[k]), '%.8f'%ITEMID_FREQ[k],'|'.join(UNITS_DICT[k]) ]) +'\n' for k in UNITS_DICT.keys()])

