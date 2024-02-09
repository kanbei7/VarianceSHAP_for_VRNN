import pandas as pd
import numpy as np
import pickle as pk
import os
import sys

'''
226566,Urine and GU Irrigant Out,Urine Volume
226627,OR Urine,Urine Volume
227489,GU Irrigant/Urine Volume Out,Urine Volume
'''
ITEM_SET = [226566, 226627, 227489]
df = pd.read_csv(os.path.join('icu','outputevents.csv'))
df = df[df['itemid'].isin(ITEM_SET)]
print('N records: %d'%len(df))
print('N stays: %d'%len(df.stay_id.unique()))
df['newname'] = ['Urine Volume']*len(df)
df.to_csv(os.path.join('outputs','urine_volume.csv'),index = False)