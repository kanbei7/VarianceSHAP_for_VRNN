import os
import sys
import json
import math
import argparse
import pandas as pd
import numpy as np
import pickle as pk
import datetime as dt
import discretization_utils as utils
from tqdm import tqdm


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--normalizer", type=str, default = "zscore", help="type of normalizer")
	parser.add_argument("--imputation", type=str, default = "forward", help="imputation method")
	parser.add_argument("--discretizer", type=str, default = "multi", help="discretization method")
	parser.add_argument("--featureset", type=str, default = "all", help="subset features")
	parser.add_argument("--binary_mask", type=str, default = "log", help="binary mask or log")
	parser.add_argument("--max_los", type=int, default = 24*30, help="max LOS in hours")
	parser.add_argument("--label", type=str, default = "death_within_hosp_basic", help="label column")

	return vars(parser.parse_args())

args = get_args()
NORMALIZER = args['normalizer'].lower()
assert(NORMALIZER in ['none', 'zscore','log','minmax','mixed'])

IMPUTATION_METHOD = args['imputation']
assert(IMPUTATION_METHOD in ['none','forward'])

DISCRETIZER = args['discretizer']
assert(DISCRETIZER in ['one','multi'])

FEAT_SET = args['featureset']
BINARY_MASK = args['binary_mask']
assert(BINARY_MASK in ['log','binary'])
LABEL_COL = args['label']

MAX_LOS = args['max_los']

str2dts = dt.datetime.strptime
date_fmt = '%Y-%m-%d'
dts_fmt = '%Y-%m-%d %H:%M:%S'

with open(os.path.join('outputs','basic_features_datatypes.json'), 'r') as f:
	DATA_TYPE_DICT = json.load(f)

FEATURES_LIST_BASIC = sorted(list(DATA_TYPE_DICT.keys()))
FEATURES_LIST_NUMERICAL = [feat for feat in FEATURES_LIST_BASIC if DATA_TYPE_DICT[feat]=='num']
FEATURES_LIST_MASKED = FEATURES_LIST_BASIC + [feat+'_mask' for feat in FEATURES_LIST_BASIC]

CAT_ENCODING = ['gcs-verbal%d'%idx for idx in range(5)] \
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

FEATURES_LIST_FULL = FEATURES_LIST_MASKED \
					+ [feat+'_disc' for feat in utils.DISCRETIZED_FEAT] \
					+ CAT_ENCODING

IMPUTED_VALUES = {}
impute_df = pd.read_csv(os.path.join('outputs','imputation.csv'))
for a,b in zip(impute_df['feat'],impute_df['value']):
	IMPUTED_VALUES[a] = float(b)

range_df = pd.read_csv(os.path.join('outputs','range_limit.csv'))
RANGE_dict = {}
for a,b in zip(range_df['feat'],range_df['max']):
	RANGE_dict[a] = min(float(b),10000)

summary_stat_df = pd.read_csv(os.path.join('outputs','summary_statistics.csv'))
statistics_dict = {}
for a,b,c,d in zip(summary_stat_df['feat'],summary_stat_df['mean'],summary_stat_df['sd'], summary_stat_df['norm_method']):
	statistics_dict[a] = {}
	statistics_dict[a]['max'] = RANGE_dict[a] if a in RANGE_dict.keys() else 10000
	statistics_dict[a]['mean'] = b
	statistics_dict[a]['sd'] = c
	statistics_dict[a]['norm_method'] = d

#obtain label and data
all_data = []
LABELS_DICT = {}
for sid in tqdm(os.listdir('stays_final')):
	ts_df = pd.read_csv(os.path.join('stays_final',sid,'aligned_flat_full.csv'))
	info_df = pd.read_csv(os.path.join('stays_final',sid,'info.csv'))
	assert(len(ts_df)>0)
	if len(ts_df)> MAX_LOS:
		continue
	sid = int(sid)
	ts_df['sid'] = [sid]*len(ts_df)
	# imputation
	if IMPUTATION_METHOD == 'zero':
		ts_df = ts_df.fillna(0)
	elif IMPUTATION_METHOD == 'forward':
		ts_df = ts_df.fillna(method="ffill")
		ts_df = ts_df.fillna(value = IMPUTED_VALUES)
	else:
		pass

	all_data.append(ts_df)
	LABELS_DICT[sid] = list(info_df[LABEL_COL])[0]
all_data = pd.concat(all_data)


#drop texts
all_data = all_data.drop(columns = utils.DISCRETIZED_FEAT)




print('Imputation completed.')

#discretization
if DISCRETIZER == 'multi':
	all_data = all_data.drop(columns = [feat+'_disc' for feat in utils.DISCRETIZED_FEAT])
elif DISCRETIZER == 'one':
	all_data = all_data.drop(columns = CAT_ENCODING)
else:
	pass


# normalize each numerical feature
def normalize_per_feature(data, feat_name, method):
	global statistics_dict
	if method == 'zscore':
		m = statistics_dict[feat_name]['mean']
		sd = statistics_dict[feat_name]['sd']
		data = [x if pd.isna(x) else (x-m)/sd for x in data]

	elif method == 'log':
		data = [x if pd.isna(x) else np.log10(x+1) for x in data]

	elif method == 'minmax':
		m = statistics_dict[feat_name]['max']
		data = [x if pd.isna(x) else x/m for x in data]

	elif method == 'mixed':

		if statistics_dict[feat_name]['norm_method'] == 'log':
			data = [x if pd.isna(x) else np.log10(x+1) for x in data]
		else:
			m = statistics_dict[feat_name]['max']
			data = [x if pd.isna(x) else x/m for x in data]

	else:
		pass

	return data

print('start normalization.')

for feat in tqdm(FEATURES_LIST_NUMERICAL):
	all_data[feat] = normalize_per_feature(list(all_data[feat]), feat, NORMALIZER)


#mask log2
print('process masks.')
if BINARY_MASK == 'log':
	for feat in tqdm(all_data.columns):
		if feat.endswith('mask'):
			all_data[feat] = all_data[feat].apply(lambda x:np.log2(x+1))

elif BINARY_MASK == 'binary':
	for feat in tqdm(all_data.columns):
		if feat.endswith('mask'):
			all_data[feat] = all_data[feat].apply(lambda x:1 if x>0 else 0)

else:
	pass


if FEAT_SET!= 'all':
	with open(os.path.join('outputs',FEAT_SET), 'r') as f:
		lines = f.readlines()
	features_final = [l.strip() for l in lines]
	all_data = all_data[['sid','hours'] + features_final]

else:
	features_final = list(all_data.columns)
	features_final.remove('sid')
	features_final.remove('hours')
for a,b in zip(all_data.columns, all_data.dtypes):
	print(a,b)
# pair with labels, drop stay id
labeled_data = [(LABELS_DICT[sid], all_data[all_data['sid']==sid].sort_values(by=['hours'], ascending = True).drop(columns = ['sid','hours']).to_numpy(), sid ) \
				for sid in all_data['sid'].unique()]


# write pkl
data_version = '_'.join([FEAT_SET.split('.')[0], LABEL_COL.split('_')[-1], NORMALIZER, DISCRETIZER, IMPUTATION_METHOD,BINARY_MASK+'mask'])
features_file_name = 'feature_names_'+ data_version + '.txt'
data_name = 'data_'+ data_version + '.pkl'
with open(os.path.join('datasets', data_name), 'wb') as f:
	pk.dump(labeled_data , f)

# write feature name
with open(os.path.join('datasets', features_file_name), 'w') as f:
	f.writelines([n + '\n' for n in features_final])
