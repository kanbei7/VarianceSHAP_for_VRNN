import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
from tqdm import tqdm

stays_df = pd.read_csv(os.path.join('outputs','pa.csv'))
stays_set = set(stays_df['stay_id'].unique())
print(len(stays_set))
'''
from perstay_raw to stays
'stay%d'%stay_id + '.csv'
'''

flst = [x for x in os.listdir('perstay_raw') if x.startswith('stay') and x.endswith('.csv')]

for fname in tqdm(flst):
	stayid = fname.split('.')[0][4:]
	if int(stayid) in stays_set:

		#make dir in stays/
		folder_name = stayid
		if  not os.path.exists(os.path.join('stays', folder_name)):
			os.mkdir(os.path.join('stays', folder_name))
		#write files
		df = pd.read_csv(os.path.join('perstay_raw',fname))
		df.to_csv(os.path.join('stays',folder_name, fname), index = False)
		#write per stay info as well
		pa_info = stays_df[stays_df['stay_id'] == int(stayid)]
		assert(len(pa_info)==1)
		pa_info.to_csv(os.path.join('stays',folder_name, 'info.csv'),index = False)		



'''
read labevents.csv by parts



'''

#lab_df1 = pd.read_csv(os.path.join('hosp','labevents.csv'), nrows = 20000)

#lab_df2 = pd.read_csv(os.path.join('hosp','labevents.csv'), skiprows = 20000, nrows = 20000)




