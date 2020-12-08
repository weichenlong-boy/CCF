# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:41:07 2020

@author: 月球来的火星人
"""
import pandas as pd

df = pd.read_csv(r'\融合文件\st_xlf.csv')
df1 = pd.read_csv(r'\融合文件\result0.8486.csv')
df2 = pd.read_csv(r'\融合文件\result0.8471.csv')
df3 = pd.read_csv(r'\融合文件\result0.8472.csv')
df0 = df.copy() 
df0['score'] = (df['score']+df3['score']+df1['score']+df2['score']*2)/5  #8532
df0.to_csv('D:/Downloads/sub_stacking_jiaquan_merge.csv', index=False)
