import numpy as np
import pickle as pk

def Chloride_mgdl2mEqL(x):
	return 10*x/35.4527

def Temperature_F2C(x):
	return (x-32)*5/9

DISCRETIZED_FEAT = [
				'gcs-verbal',
				'gcs-eye',
				'gcs-motor',
				'braden-sense',
				'braden-nut',
				'braden-moist',
				'braden-mob',
				'braden-fric',
				'braden-act',
				'temp site',
				'urine color'
				]



def convert_GCS_Verbal(x):
	value = 5
	vec = [0]*5
	x=x.strip().lower()
	if 'et' in x:
		return 1, [0,0,0,1,0]
	if 'no response' in x or 'noresponse' in x:
		return 1, [0,0,0,0,1]
	if 'incomprehensible' in x:
		return 2, [0,0,1,0,0]
	if 'inappropriate' in x:
		return 3, [0,1,0,0,0]
	if 'confuse' in x:
		return 4, [1,0,0,0,0]

	return value, vec


def convert_GCS_Eye(x):
	value = 4
	vec = [0]*3
	x=x.strip().lower()
	if 'none' in x:
		return 1, [0,0,1]
	if 'pain' in x:
		return 2, [0,1,0]
	if 'speech' in x:
		return 3, [1,0,0]

	return value, vec

def convert_GCS_Motor(x):
	value = 6
	vec = [0]*5
	x=x.strip().lower()
	if 'no response' in x or 'noresponse' in x:
		return 1, [0,0,0,0,1]
	if 'extension' in x:
		return 2, [0,0,0,1,0]
	if 'withdraw' in x:
		return 4, [0,1,0,0,0]
	if 'flexion' in x and 'abnormal' in x:
		return 3, [0,0,1,0,0]
	if 'pain' in x:
		return 5, [1,0,0,0,0]

	return value, vec



def convert_Braden_Sense(x):
	value = 4
	vec = [0]*3
	x=x.strip().lower()
	if 'complete' in x and 'limit' in x:
		return 1, [0,0,1]
	if 'very' in x and 'limit' in x:
		return 2, [0,1,0]
	if 'slight' in x and 'impair' in x:
		return 3, [1,0,0]

	return value, vec



def convert_Braden_Nut(x):
	value = 4
	vec = [0]*3
	x=x.strip().lower()
	if 'poor' in x:
		return 1, [0,0,1]
	if 'inadequate' in x:
		return 2, [0,1,0]
	if 'adequate' in x:
		return 3, [1,0,0]

	return value, vec



def convert_Braden_Moist(x):
	x=x.strip().lower()
	if 'rarely' in x :
		return 4, [0,0,0]
	if 'consistent' in x:
		return 1, [0,0,1]
	if 'occasional' in x:
		return 3, [1,0,0]

	return 2, [0,1,0]



def convert_Braden_Mob(x):
	value = 4
	vec = [0]*3
	x=x.strip().lower()
	if 'complete' in x:
		return 1, [0,0,1]
	if 'very' in x:
		return 2, [0,1,0]
	if 'slight' in x:
		return 3, [1,0,0]

	return value, vec


def convert_Braden_Fric(x):
	x=x.strip().lower()
	if 'no' in x and 'problem' in x:
		return 3, [0,0]
	if 'potential' in x:
		return 2, [1,0]

	return 1, [0,1]



def convert_Braden_Act(x):
	value = 4
	vec = [0]*3
	x=x.strip().lower()
	if 'bed' in x:
		return 1, [0,0,1]
	if 'chair' in x:
		return 2, [0,1,0]
	if 'occasional' in x:
		return 3, [1,0,0]

	return value, vec


def convert_Temp_Site(x):
	x=x.strip().lower()
	if 'oral' in x:
		return 0, [0,0,0,0,0,0]
	if 'axillary' in x:
		return 1, [0,0,0,0,0,1]
	if 'blood' in x:
		return 2, [0,0,0,0,1,0]
	if 'rectal' in x:
		return 3, [0,0,0,1,0,0]
	if 'esophog' in x:
		return 4, [0,0,1,0,0,0]
	if 'tympan' in x:
		return 5, [0,1,0,0,0,0]

	return 6, [1,0,0,0,0,0] 

def convert_Urine_Color(x):
	x=x.strip().lower()
	if 'yellow' in x and 'light' in x:
		return 0, [0,0,0,0,0,0,0,0]
	if 'green' in x:
		return 8, [1,0,0,0,0,0,0,0]
	if 'brown' in x:
		return 7, [0,1,0,0,0,0,0,0]
	if 'icteric' in x:
		return 4, [0,0,0,0,1,0,0,0]
	if 'amber' in x:
		return 2, [0,0,0,0,0,0,1,0]
	if 'orange' in x:
		return 3, [0,0,0,0,0,1,0,0]
	if 'pink' in x:
		return 5, [0,0,0,1,0,0,0,0]
	if 'red' in x:
		return 6, [0,0,1,0,0,0,0,0]

	return 1, [0,0,0,0,0,0,0,1]


