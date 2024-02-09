import os
import sys
import json
import math
import pandas as pd
import numpy as np
import pickle as pk
import datetime as dt
import discretization_utils as utils
from tqdm import tqdm


str2dts = dt.datetime.strptime
date_fmt = '%Y-%m-%d'
dts_fmt = '%Y-%m-%d %H:%M:%S'

def calc_timediff_hrs(t1,t2):
	t1 = str2dts(t1, dts_fmt)
	t2 = str2dts(t2, dts_fmt)
	return max(0,math.ceil((t1-t2).total_seconds()/3600))

def last_valid(ts):
	N = len(ts)
	for i in range(len(ts)):
		x=ts[N-1-i]
		if not pd.isna(x):
			return x
	return ts[-1]

with open(os.path.join('outputs','basic_features_datatypes.json'), 'r') as f:
	DATA_TYPE_DICT = json.load(f)

FEATURES_LIST_BASIC = sorted(list(DATA_TYPE_DICT.keys()))
FEATURES_LIST_NUMERICAL = [feat for feat in FEATURES_LIST_BASIC if DATA_TYPE_DICT[feat]=='num']
FEATURES_LIST_MASKED = FEATURES_LIST_BASIC + [feat+'_mask' for feat in FEATURES_LIST_BASIC]
'''
FEATURES_LIST_FULL = FEATURES_LIST_MASKED \
					+ [feat+'_disc' for feat in utils.DISCRETIZED_FEAT] \
					+ ['gcs_verbal%d'%idx for idx in range(5)] \
					+ ['gcs_eye%d'%idx for idx in range(3)] \
					+ ['gcs_motor%d'%idx for idx in range(5)] \
					+ ['braden_sense%d'%idx for idx in range(3)] \
					+ ['braden_nut%d'%idx for idx in range(3)] \
					+ ['braden_moist%d'%idx for idx in range(3)] \
					+ ['braden_mob%d'%idx for idx in range(3)] \
					+ ['braden_fric%d'%idx for idx in range(2)] \
					+ ['braden_act%d'%idx for idx in range(3)] \
					+ ['temp site%d'%idx for idx in range(6)] \
					+ ['urine color%d'%idx for idx in range(8)]
'''

for sid in tqdm(os.listdir('stays_final')[:15000]):
	ts_df = pd.read_csv(os.path.join('stays_final',sid,'timeseries.csv'))
	ts_df.drop_duplicates(subset=['newname', 'time'], keep='last', inplace = True, ignore_index = True)

	info_df = pd.read_csv(os.path.join('stays_final',sid,'info.csv'))
	intime = list(info_df['intime'])[0]

	#flattern, add missing feat
	flat_df = ts_df.pivot(index = 'time', columns = 'newname', values = 'value')

	missing_feat = [feat for feat in FEATURES_LIST_BASIC if not feat in flat_df.columns]
	flat_df = flat_df.reindex(columns = flat_df.columns.tolist() + missing_feat)
	
	#reorder
	flat_df.reset_index(inplace = True)
	flat_df = flat_df[['time']+FEATURES_LIST_BASIC]

	#check float() for numerical features
	for feat in FEATURES_LIST_NUMERICAL:
		if len(flat_df[flat_df[feat].notnull()])>0:
			check = [float(x) for x in flat_df[feat] if not pd.isna(x)]

	#add hour label
	flat_df['hrsSinceAdmission'] = [calc_timediff_hrs(t,intime) for t in flat_df['time']]
	max_hrs = max(flat_df['hrsSinceAdmission'])

	#align to hours
	#add imp mask/num of measurements in this hours
	aligned_df = []
	for hrs, grp_raw in flat_df.groupby('hrsSinceAdmission'):
		grp = grp_raw.copy()
		grp.sort_values(by = ['time'], ascending = True, inplace=True, ignore_index = True)
		ctnt = [hrs] + [last_valid(grp[feat].tolist()) if len(grp[grp[feat].notnull()])>0 else np.nan for feat in FEATURES_LIST_BASIC] + [len(grp[grp[feat].notnull()])  for feat in FEATURES_LIST_BASIC]
		aligned_df.append(ctnt)
	#fill up blank hours between min-max
	for hrs in [h for h in range(max_hrs) if not h in flat_df['hrsSinceAdmission'].unique()]:
		aligned_df.append([hrs] + [np.nan for _ in range(len(FEATURES_LIST_MASKED))])

	aligned_df = pd.DataFrame(data = aligned_df, columns = ['hours']+FEATURES_LIST_MASKED)
	aligned_df.sort_values(by = ['hours'], ascending = True, inplace=True, ignore_index = True)
	
	#reorder
	aligned_df = aligned_df[['hours']+FEATURES_LIST_MASKED]
	
	#write
	aligned_df.to_csv(os.path.join('stays_final',sid,'aligned_flat.csv'), index = False)
