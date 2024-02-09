import numpy as np
import pandas as pd
import pickle as pk
import datetime as dt
import os
from tqdm import tqdm

str2dts = dt.datetime.strptime
date_fmt = '%Y-%m-%d'
dts_fmt = '%Y-%m-%d %H:%M:%S'
icu_stay_cols = ['subject_id','hadm_id','stay_id','first_careunit','last_careunit','intime','outtime','los']
pa_cols = ['subject_id','gender','anchor_age','anchor_year','dod']
adm_cols = ['subject_id','hadm_id','admittime','dischtime','deathtime','admission_type','admission_location','discharge_location','race','hospital_expire_flag']

stay_file = os.path.join('outputs','icustays-selected.csv')
stays_df = pd.read_csv(stay_file)
stays_df = stays_df[icu_stay_cols]

pa_file = os.path.join('hosp','patients.csv')
pa_df = pd.read_csv(pa_file)
pa_df = pa_df[pa_cols]

adm_file = os.path.join('hosp','admissions.csv')
adm_df = pd.read_csv(adm_file)
adm_df = adm_df[adm_cols]

'''
[outputs/icustays-selected.csv]
subject_id,hadm_id,stay_id,
first_careunit,last_careunit,
intime,outtime,los
yyyy-mm-dd h:m:s

[hosp/patients.csv]
subject_id,gender,anchor_age,
anchor_year,dod
yyyy-mm-dd

[hosp/admissions.csv]
subject_id,hadm_id,admittime,
dischtime,deathtime,race,hospital_expire_flag

'''

#intersection of pid in stays and pa
pa_pid = set(pa_df.subject_id.unique())
pa_set = set([pid  for pid in stays_df.subject_id.unique() if pid in pa_pid])

#subset all by subjectid
stays_df = stays_df[stays_df.subject_id.isin(pa_set)]
pa_df = pa_df[pa_df.subject_id.isin(pa_set)]
adm_df = adm_df[adm_df.subject_id.isin(pa_set)]

#in,out, dischtime, admittime notnull
stays_df =stays_df[stays_df['outtime'].notnull()]
stays_df =stays_df[stays_df['outtime'].notnull()]
adm_df = adm_df[adm_df['admittime'].notnull()]
adm_df = adm_df[adm_df['dischtime'].notnull()]
#join stay and pa
pa_df.drop_duplicates(inplace=True)
pa_df.reset_index(drop = True,inplace=True)
stays_df = stays_df.merge(pa_df , how = 'left', on = ['subject_id'])
stays_df.drop_duplicates(inplace=True)
stays_df.reset_index(drop = True,inplace=True)

#intersection of hadmid
hadmid_stay = set(stays_df.hadm_id.unique())
hadmid_set = set([hid  for hid in adm_df.hadm_id.unique() if hid in hadmid_stay])

#subset by hadmid
stays_df = stays_df[stays_df.hadm_id.isin(hadmid_set)]
adm_df = adm_df[adm_df.hadm_id.isin(hadmid_set)]

#drop duplicates
adm_df.drop_duplicates(inplace=True)
adm_df.reset_index(drop = True,inplace=True)
stays_df = stays_df.merge(adm_df, how = 'left', on = ['subject_id','hadm_id'])
stays_df = stays_df[(stays_df['intime'].notnull()) & (stays_df['outtime'].notnull()) & (stays_df['admittime'].notnull()) & (stays_df['dischtime'].notnull())]
stays_df.drop_duplicates(inplace=True)
stays_df.reset_index(drop = True,inplace=True)
#impute race?
print('N PA: %d'%(len(stays_df.subject_id.unique())))
print('N Adm: %d'%(len(stays_df.hadm_id.unique())))
print('N Stays: %d'%(len(stays_df.stay_id.unique())))

print('Missing Race: %d'%(len(stays_df) - len(stays_df[stays_df['race'].notnull()])))
print('ICU Death Discharge: %d'%(len(stays_df[stays_df['dod'].notnull()].subject_id.unique())))
print('IHM Deaths: ')
print(stays_df['hospital_expire_flag'].value_counts())

#time to dates, calc differences
'''
outtime - dod
outtime - deathtime
admittime - dod
admittime - deathtime
dischtime - dod
dischtime - deathtime


dischtime >= admittime
dischtime >= outtime

dod.notnull but deathtime is null
dod.notnull but hospital_expire_flag = 0
deathtime but hospital_expire_flag = 0
'''
def date_diff(t1,t2):
	if pd.isna(t1) or pd.isna(t2):
		return -1
	t1 = str2dts(t1, date_fmt)
	t2 = str2dts(t2, date_fmt)
	return (t1-t2).days

outdates = stays_df['outtime'].apply(lambda x:x.split(' ')[0])
admdates = stays_df['admittime'].apply(lambda x:x.split(' ')[0])
dschrgdates = stays_df['dischtime'].apply(lambda x:x.split(' ')[0])

print('Adm time > Dschrg time: %d'%(len([date_diff(x,y) for x,y in zip(admdates, dschrgdates) if date_diff(x,y)>0])))
print('Adm time > Outtime: %d'%(len([date_diff(x,y) for x,y in zip(admdates, outdates) if date_diff(x,y)>0])))



death_df = stays_df[stays_df['deathtime'].notnull()]
outdates = death_df['outtime'].apply(lambda x:x.split(' ')[0])
admdates = death_df['admittime'].apply(lambda x:x.split(' ')[0])
dschrgdates = death_df['dischtime'].apply(lambda x:x.split(' ')[0])
deathdates = death_df['deathtime'].apply(lambda x:x.split(' ')[0])
print('Outtime > deathtime: %d'%(len([date_diff(x,y) for x,y in zip(outdates, deathdates) if date_diff(x,y)>0])))
print('Dschrg time > deathtime: %d'%(len([date_diff(x,y) for x,y in zip(dschrgdates, deathdates) if date_diff(x,y)>0])))
print('Adm time > deathtime: %d'%(len([date_diff(x,y) for x,y in zip(admdates, deathdates) if date_diff(x,y)>0])))
'''
exclude anomalies
Adm time > dod: 4

truncate to dod or deathtime
Outtime > deathtime: 580

How many left? discard 
Outtime > dod: 623
Dschrg time > dod: 9


'''
print(len(stays_df))
adm_leq_dod = pd.Series([date_diff(x,y)<=0 for x,y in zip(stays_df['admittime'].apply(lambda x:x.split(' ')[0]), stays_df.dod)])
stays_df = stays_df[adm_leq_dod]
stays_df.reset_index(drop = True,inplace=True)
print(len(stays_df))

#clamp t2 to t1
#t1 is dod or deathtime
def date_clamp(t1,t2):
	if pd.isna(t1) or pd.isna(t2):
		return t2

	t2 = str2dts(t2, dts_fmt)
	t1 = str2dts(t1, dts_fmt) if len(t1.split(' '))>1 else str2dts(t1, date_fmt)
	
	if (t2-t1).total_seconds()>0:
		return t1.strftime(dts_fmt)
	return t2

stays_df['outtime'] = [date_clamp(a,b) for a,b in zip(stays_df['deathtime'], stays_df['outtime'])]

dod_df = stays_df[stays_df['dod'].notnull()]
outdates = dod_df['outtime'].apply(lambda x:x.split(' ')[0])
admdates = dod_df['admittime'].apply(lambda x:x.split(' ')[0])
dschrgdates = dod_df['dischtime'].apply(lambda x:x.split(' ')[0])

#'hospital_expire_flag'
print('Outtime > dod: %d'%(len([date_diff(x,y) for x,y in zip(outdates, dod_df.dod) if date_diff(x,y)>0])))
print('#hosp expire: %d'%(sum([z for x,y,z in zip(outdates, dod_df.dod, dod_df.hospital_expire_flag) if date_diff(x,y)>0] )))
print('Dschrg time > dod: %d'%(len([date_diff(x,y) for x,y in zip(dschrgdates, dod_df.dod) if date_diff(x,y)>0])))
print('#hosp expire: %d'%(sum( [z for x,y,z in zip(dschrgdates, dod_df.dod, dod_df.hospital_expire_flag) if date_diff(x,y)>0])))

print('Adm time > dod: %d'%(len([date_diff(x,y) for x,y in zip(admdates, dod_df.dod) if date_diff(x,y)>0])))

#drop abnormal patients
print('N PA: %d'%(len(stays_df.subject_id.unique())))
print('N Adm: %d'%(len(stays_df.hadm_id.unique())))
print('N Stays: %d'%(len(stays_df.stay_id.unique())))
outtime_g_dod = pd.Series([date_diff(x,y)>0 for x,y in zip(stays_df['outtime'].apply(lambda x:x.split(' ')[0]), stays_df.dod)])
dischtime_g_dod = pd.Series([date_diff(x,y)>0 for x,y in zip(stays_df['dischtime'].apply(lambda x:x.split(' ')[0]), stays_df.dod)])
print(len(outtime_g_dod ))
print(len(dischtime_g_dod))

abnormal_pa = set(list(stays_df[outtime_g_dod].subject_id.unique()) + list(stays_df[dischtime_g_dod].subject_id.unique()))

stays_df = stays_df[~stays_df.subject_id.isin(abnormal_pa)]
stays_df.reset_index(drop = True,inplace=True)
print('Abnormal pa dropped.')
print('N PA: %d'%(len(stays_df.subject_id.unique())))
print('N Adm: %d'%(len(stays_df.hadm_id.unique())))
print('N Stays: %d'%(len(stays_df.stay_id.unique())))
'''
has dod, but no deathtime
has deathtime but no dod
if both, are they consistent?

dod notnull but deathtime = null
dod null but deathtime notnull
dod and deathtime both available
'''


check_df1 = stays_df[(stays_df['dod'].notnull()) & (stays_df['deathtime'].isnull())].copy()
print('Dod notnull but death time is null: %d'%len(check_df1))
check_df2 = stays_df[(stays_df['dod'].isnull()) & (stays_df['deathtime'].notnull())]
print('Dod is null but death time notnull: %d'%len(check_df2))
check_df3 = stays_df[(stays_df['dod'].notnull()) & (stays_df['deathtime'].notnull())]
print('Dod notnull and death time notnull: %d'%len(check_df3))
'''
check_df1.to_csv(os.path.join('outputs','check1.csv'),index = False)
check_df1.sort_values(by='admittime', ascending=True, inplace = True)
check_df1.reset_index(drop = True, inplace=True)
check_df1.drop_duplicates(subset=['subject_id'], keep='last', inplace = True)

check_df1.to_csv(os.path.join('outputs','check1_last.csv'),index = False)
print(len(check_df1['subject_id'].unique()))
'''

'''
age at admission
-Note that ages 89 or above are still 91
- Truncated to 90
- use anchor_year
- create admission year
'''
stays_df['admission_year'] = stays_df['admittime'].apply(lambda x:int(x.split(' ')[0].split('-')[0]))
#print( stays_df['admission_year'].describe())

def calc_age(anc_age, anc_yr, adm_yr):
	if anc_age>89:
		return 90
	return np.clip(anc_age + max(0, int(adm_yr) - int(anc_yr)), 0, 90)

stays_df['age_at_admission'] = [ calc_age(a,b,c) for a,b,c in zip(stays_df['anchor_age'], stays_df['anchor_year'], stays_df['admission_year'])]

print('N PA <=52: %d'%len(stays_df[stays_df['age_at_admission']<=52].subject_id.unique()))
print('N PA 52< <65: %d'%len(stays_df[(stays_df['age_at_admission']>52)  & (stays_df['age_at_admission']<65)  ].subject_id.unique()))
print('N PA >=65: %d'%len(stays_df[stays_df['age_at_admission']>=65].subject_id.unique()))
'''
# death labels
- add hospice column (hadm_id)
	Hospice.pkl

- add DNR column (hadm_id)
	DNR.pkl

- create basic death_within_hosp labels
	- if hosp_expire_flag>0
	- if deathtime notnull
	- in the discharge death dictionary
		Discharge_death.pkl (hadm_id)
		discharged died but no deathtime?
	- if discharged to hospice or had hospice diagnosis code

- create full labels
	- basic labels
	- if DNR order issued
'''
with open(os.path.join('outputs','DNR.pkl'),'rb') as f:
	dnr_dict = pk.load(f)

with open(os.path.join('outputs','Hospice.pkl'),'rb') as f:
	hospice_dict = pk.load(f)

with open(os.path.join('outputs','Discharge_death.pkl'),'rb') as f:
	discharge_DIED_dict = pk.load(f)


stays_df['DNR'] = stays_df['hadm_id'].apply(lambda x:1 if x in dnr_dict.keys() else 0)
stays_df['HOSPICE'] = stays_df['hadm_id'].apply(lambda x:1 if x in hospice_dict.keys() else 0)
stays_df['DSCHRG_LOC_DIED'] = stays_df['hadm_id'].apply(lambda x:1 if x in discharge_DIED_dict.keys() else 0)
stays_df['deathtime_notnull'] = stays_df['deathtime'].apply(lambda x:0 if pd.isna(x) else 1)
stays_df['death_within_hosp_basic'] = [min(1, sum([a,b,c,d])) for a,b,c,d in zip(stays_df['HOSPICE'], stays_df['hospital_expire_flag'],stays_df['DSCHRG_LOC_DIED'], stays_df['deathtime_notnull']) ]

#check how many DNR pa death_within_hosp_basic==1
print('DNR death rate(#PA): %.4f'%(len(stays_df[(stays_df['DNR']>0) & (stays_df['death_within_hosp_basic']>0)].subject_id.unique())/len(stays_df[stays_df['DNR']>0].subject_id.unique())))
print('DNR death rate(#hadm): %.4f'%(len(stays_df[(stays_df['DNR']>0) & (stays_df['death_within_hosp_basic']>0)].hadm_id.unique())/len(stays_df[stays_df['DNR']>0].hadm_id.unique())))

#full label
stays_df['death_within_hosp_full'] = (stays_df['death_within_hosp_basic'] + stays_df['DNR']).apply(lambda x:min(1,x))


#print(stays_df['death_within_hosp_basic'].value_counts())


# LOS in hours (ceiling)

stays_df['LOS_hrs'] =  stays_df['los']*24
print(stays_df['LOS_hrs'].describe())
#print(stays_df['race'].value_counts())

def grouprace(x):
	if 'ASIAN' in x:
		return 'ASIAN'
	elif 'BLACK' in x:
		return 'BLACK'
	elif 'HISPANIC' in x:
		return 'HISPANIC'
	elif 'WHITE' in x:
		return 'WHITE'
	else:
		return 'OTHERS'


stays_df['race_grouped'] = stays_df['race'].apply(grouprace)
stays_df.to_csv(os.path.join('outputs','pa.csv'),index = False)
print(stays_df.columns)
pa_df = stays_df.drop_duplicates(subset=['subject_id'], keep = 'last')
print(len(pa_df))
print(pa_df['race_grouped'].value_counts())


