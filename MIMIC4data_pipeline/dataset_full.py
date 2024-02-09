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

with open(os.path.join('outputs','basic_features_datatypes.json'), 'r') as f:
	DATA_TYPE_DICT = json.load(f)

FEATURES_LIST_BASIC = sorted(list(DATA_TYPE_DICT.keys()))
FEATURES_LIST_NUMERICAL = [feat for feat in FEATURES_LIST_BASIC if DATA_TYPE_DICT[feat]=='num']
FEATURES_LIST_MASKED = FEATURES_LIST_BASIC + [feat+'_mask' for feat in FEATURES_LIST_BASIC]

FEATURES_LIST_FULL = FEATURES_LIST_MASKED \
					+ [feat+'_disc' for feat in utils.DISCRETIZED_FEAT] \
					+ ['gcs-verbal%d'%idx for idx in range(5)] \
					+ ['gcs-eye%d'%idx for idx in range(3)] \
					+ ['gcs-motor%d'%idx for idx in range(5)] \
					+ ['braden-sense%d'%idx for idx in range(3)] \
					+ ['braden-nut%d'%idx for idx in range(3)] \
					+ ['braden-moist%d'%idx for idx in range(3)] \
					+ ['braden-mob%d'%idx for idx in range(3)] \
					+ ['braden-fric%d'%idx for idx in range(2)] \
					+ ['braden-act%d'%idx for idx in range(3)] \
					+ ['temp site%d'%idx for idx in range(6)] \
					+ ['urine color%d'%idx for idx in range(8)]

discretization_functions = {
	"gcs-verbal":utils.convert_GCS_Verbal,
	"gcs-eye":utils.convert_GCS_Eye,
	"gcs-motor":utils.convert_GCS_Motor,
	"braden-sense":utils.convert_Braden_Sense,
	"braden-nut":utils.convert_Braden_Nut,
	"braden-moist":utils.convert_Braden_Moist,
	"braden-mob":utils.convert_Braden_Mob,
	"braden-fric":utils.convert_Braden_Fric,
	"braden-act":utils.convert_Braden_Act,
	"temp site":utils.convert_Temp_Site,
	"urine color":utils.convert_Urine_Color,
}

discretization_dims = {
	"gcs-verbal":5,
	"gcs-eye":3,
	"gcs-motor":5,
	"braden-sense":3,
	"braden-nut":3,
	"braden-moist":3,
	"braden-mob":3,
	"braden-fric":2,
	"braden-act":3,
	"temp site":6,
	"urine color":8,
}


def discretize(data, feat, fn, L):
	encoding_df = []
	numerical_encoding = []
	for t in data[feat].tolist():
		if not pd.isna(t):
			d, encoding_vec = fn(t)
			encoding_df.append(encoding_vec)
			numerical_encoding.append(d)
		else:
			encoding_df.append([np.nan for _ in range(L)])
			numerical_encoding.append(np.nan)

	encoding_df = pd.DataFrame(data = encoding_df, columns = [feat+'%d'%idx for idx in range(L)])
	return numerical_encoding, encoding_df


for sid in tqdm(os.listdir('stays_final')):
	ts_df = pd.read_csv(os.path.join('stays_final',sid,'aligned_flat.csv'))
	assert(len(ts_df)>0)
	info_df = pd.read_csv(os.path.join('stays_final',sid,'info.csv'))

	#discretize
	for feat in discretization_functions.keys():
		fn = discretization_functions[feat]
		encoding_lengths = discretization_dims[feat]
		categories, encoding_df = discretize(ts_df, feat, fn, encoding_lengths)
		ts_df[feat+'_disc'] = categories
		ts_df = pd.concat([ts_df, encoding_df], axis = 1)

	#write
	ts_df = ts_df[['hours'] + FEATURES_LIST_FULL]
	ts_df.to_csv(os.path.join('stays_final',sid,'aligned_flat_full.csv'), index = False)


# summary statistics of all numerical features
DATA_DICT = {feat:[] for feat in FEATURES_LIST_NUMERICAL}


for sid in tqdm(os.listdir('stays_final')):
	ts_df = pd.read_csv(os.path.join('stays_final',sid,'aligned_flat_full.csv'))
	
	for feat in FEATURES_LIST_NUMERICAL:
		tmp_df = ts_df[ts_df[feat].notnull()]
		if len(tmp_df)>0:
			DATA_DICT[feat].extend(tmp_df[feat].tolist())

stat_df = []
stat_dict = {}
for feat in FEATURES_LIST_NUMERICAL:
	s = DATA_DICT[feat]
	m1 = np.mean(s)
	m2 = max(s)
	m3 = min(s)
	m4 = np.median(s)
	m5 = np.std(s)
	N = len(s)
	n_above = len([x for x in s if x>10000])
	n_below = len([x for x in s if x<0])
	stat_dict[feat] = {}
	stat_dict[feat]['mean'] = m1
	stat_dict[feat]['max'] = m2
	stat_dict[feat]['min'] = m3
	stat_dict[feat]['median'] = m4
	stat_dict[feat]['sd'] = m5

	stat_df.append([feat,m1,m2,m3,m4,m5,N,n_above, n_below])

stat_df = pd.DataFrame(data = stat_df, columns = ['feat', 'mean', 'max', 'min', 'median', 'sd','N','N-above','N-below'])
stat_df.to_csv(os.path.join('outputs','summary_statistics.csv'),index=False)
with open(os.path.join('outputs','summary_stat.pkl'),'wb') as f:
	pk.dump(stat_dict,f)
# normalization/standardization