# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:48:24 2020

@author: mohdf
"""

import pandas as pd 
from sklearn.utils import shuffle

def shuffle_data(df, s):
	#shuffles the targets and data together	to keep corresponding index values constant
	df['target'] = s
	df = shuffle(df)
	s = df.target
	del df['target']
	return df, s 