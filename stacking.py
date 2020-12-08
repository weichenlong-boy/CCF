# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:26:46 2020

@author: 月球来的火星人
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

cat_8447train = pd.read_csv('../input/stacking-file/cat_0.864651train.csv')
cat_8417train = pd.read_csv('../input/stacking-file/cat_8417train.csv')
lgb_8425train = pd.read_csv('../input/53425324/lgb_train_data_01.csv')
cat_8447test = pd.read_csv('../input/stacking-file/cat_0.864651test.csv')
cat_8417test = pd.read_csv('../input/stacking-file/cat_8417test.csv')
lgb_8425test = pd.read_csv('../input/53425324/lgb_test_data_01.csv')
entprise_info = pd.read_csv('risk-prediction-of-illegal/entprise_info.csv')
cat_8447train = cat_8447train.merge(entprise_info, how='left')

train = pd.concat([cat_8447train[['score','label']],cat_8417train['score'],lgb_8425train['score']],axis=1)
test = pd.concat([cat_8447test['score'],cat_8417test['score'],lgb_8425test['score']],axis=1)
test.columns = ['cat_8447train','cat_8417train','lgb_8425train']
train.columns = ['cat_8447train','label','cat_8417train','lgb_8425train']
col = [i for i in train.columns if i!= 'label']

xlf=xgb.XGBClassifier(max_depth=5,learning_rate=0.05,n_estimators=55,reg_alpha=0.005,n_jobs=8,random_state=2020,importance_type='total_cover')
xlf.fit(train[col], train['label'])
pred_xlf = xlf.predict_proba(test[col])[:,1]
print('f1_score:',f1_score(xlf.predict_proba(train[col])[:,1].round(),train['label']))

st = cat_8447test[['id']]
st['score'] = pred_xlf
st.to_csv('st_xlf.csv', index=False)
