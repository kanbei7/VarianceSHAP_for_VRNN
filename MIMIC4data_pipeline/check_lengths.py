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
str2dts = dt.datetime.strptime
date_fmt = '%Y-%m-%d'
dts_fmt = '%Y-%m-%d %H:%M:%S'

def calc_timediff_hrs(t1,t2):
	t1 = str2dts(t1, dts_fmt)
	t2 = str2dts(t2, dts_fmt)
	return max(0,math.ceil((t1-t2).total_seconds()/3600))

res = []
for sid in tqdm(os.listdir('stays_final')):
	ts_df = pd.read_csv(os.path.join('stays_final',sid,'aligned_flat_full.csv'))
	res.append(len(ts_df))
	#ts_df = pd.read_csv(os.path.join('stays_final',sid,'timeseries.csv'))

	#info_df = pd.read_csv(os.path.join('stays_final',sid,'info.csv'))
	#intime = list(info_df['intime'])[0]

	#los = max([calc_timediff_hrs(t,intime) for t in ts_df['time']])

	#res.append(los)

print(pd.Series(res).describe())