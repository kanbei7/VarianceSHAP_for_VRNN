import pandas as pd
import numpy as np
import pickle as pk
import os
from tqdm import tqdm
'''
220561,ZINR
227467, INR

226981, Albumin_ApacheIV
220574, ZAlbumin
227456, Albumin


228389,Sodium (serum) (soft)
227052,Sodium_ApacheIV
220645,Sodium (serum)
'''

ZINR = []
INR = []

ALBM = []
ZALBM  = []
ALBM_APACHE =[]

SODIUM_SOFT = []
SODIUM_APACHE = []
SODIUM = []


for folder in tqdm(os.listdir('stays_raw')):
	df = pd.read_csv(os.path.join('stays_raw',folder, 'stay'+folder + '.csv'))

	if len(df[df.itemid == 220561])>0:
		ZINR.append(df[df.itemid == 220561].copy())
	if len(df[df.itemid == 227467])>0:
		INR.append(df[df.itemid == 227467].copy())
	
	if len(df[df.itemid == 226981])>0:
		ALBM.append(df[df.itemid == 226981].copy())
	if len(df[df.itemid == 220574])>0:
		ZALBM.append(df[df.itemid == 220574].copy())
	if len(df[df.itemid == 227456])>0:
		ALBM_APACHE.append(df[df.itemid == 227456].copy())
	
	if len(df[df.itemid == 228389])>0:
		SODIUM_SOFT.append(df[df.itemid == 228389].copy())
	if len(df[df.itemid == 227052])>0:
		SODIUM_APACHE.append(df[df.itemid == 227052].copy())
	if len(df[df.itemid == 220645])>0:
		SODIUM.append(df[df.itemid == 220645].copy())


print(len(ZINR))
print(len(INR))

print(len(ALBM))
print(len(ZALBM))
print(len(ALBM_APACHE))

print(len(SODIUM_SOFT))
print(len(SODIUM_APACHE))
print(len(SODIUM))


if len(ZINR)>0:
	ZINR = pd.concat(ZINR)
	ZINR.reset_index(drop = True,inplace=True)
	ZINR.to_csv(os.path.join('outputs','ZINR.csv'),index = False)

if len(INR)>0:
	INR = pd.concat(INR)
	INR.reset_index(drop = True,inplace=True)
	INR.to_csv(os.path.join('outputs','INR.csv'),index = False)



if len(ALBM)>0:
	ALBM = pd.concat(ALBM)
	ALBM.reset_index(drop = True,inplace=True)
	ALBM.to_csv(os.path.join('outputs','ALBM.csv'),index = False)

if len(ZALBM)>0:
	ZALBM  = pd.concat(ZALBM)
	ZALBM.reset_index(drop = True,inplace=True)
	ZALBM.to_csv(os.path.join('outputs','ZALBM.csv'),index = False)

if len(ALBM_APACHE)>0:
	ALBM_APACHE = pd.concat(ALBM_APACHE)
	ALBM_APACHE.reset_index(drop = True,inplace=True)
	ALBM_APACHE.to_csv(os.path.join('outputs','ALBM_APACHE.csv'),index = False)




if len(SODIUM_SOFT)>0:
	SODIUM_SOFT = pd.concat(SODIUM_SOFT)
	SODIUM_SOFT.reset_index(drop = True,inplace=True)
	SODIUM_SOFT.to_csv(os.path.join('outputs','SODIUM_SOFT.csv'),index = False)

if len(SODIUM_APACHE)>0:
	SODIUM_APACHE = pd.concat(SODIUM_APACHE)
	SODIUM_APACHE.reset_index(drop = True,inplace=True)
	SODIUM_APACHE.to_csv(os.path.join('outputs','SODIUM_APACHE.csv'),index = False)

if len(SODIUM)>0:
	SODIUM = pd.concat(SODIUM)
	SODIUM.reset_index(drop = True,inplace=True)
	SODIUM.to_csv(os.path.join('outputs','SODIUM.csv'),index = False)
