# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:19:23 2020

@author: 月球来的火星人
"""
# =============================================================================
# lightgbm版本2.3.0
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import matplotlib.pylab as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def deal_df1():
    data = pd.read_csv('../data/train/base_info.csv')
#     data['opscope'] = data['opscope'].apply(lambda x: x.split('（依法须经批准的项目，经相关部门批准后方可开展经营活动）')[0])
    data['opfrom'] = pd.to_datetime(data['opfrom'])
    data['opto'] = pd.to_datetime(data['opto'])
    data['use_time'] = (data['opto'] - data['opfrom']).dt.days
#     data['rate'] = data['reccap'] / data['regcap']
    data['opform'].fillna('0',inplace=True)
    Dis_cols = ['oplocdistrict','industryphy','industryco','dom','enttype','enttypeitem','state',
           'orgid','jobid','opform','enttypeminu','protype','oploc','enttypegb']
    for f in tqdm(Dis_cols):
        le=LabelEncoder()
        data[f]=le.fit_transform(data[f])
#     data = data.merge(gen_user_tfidf_features(df=data, value='opscope',n=3), on=['id'], how='left')
#     data = data.merge(gen_user_countvec_features(df=data, value='opscope',n=3), on=['id'], how='left')
    data.drop(['opfrom','opto','ptbusscope','midpreindcode','opscope','parnum','congro',
              'forreccap'],axis=1,inplace=True)
    return data


def deal_df5():
    df5 = pd.read_csv('../data/train/news_info.csv')
    df_create = df5.groupby(['id'])['public_date'].count().reset_index()
    df_create = df_create.merge(get_tfidf_features(df=df5, value='positive_negtive',n=2), on=['id'], how='left')
    df_create = df_create.merge(get_countvec_features(df=df5, value='positive_negtive',n=2), on=['id'], how='left')
    del df_create['public_date']
    return df_create

df1 = deal_df1()
df5 = deal_df5()
label = pd.read_csv('../data/train/entprise_info.csv')

data = df1.merge(label,on=['id'],how='left')
data = data.merge(df5,on='id',how='left')
# data.to_csv('fea_data/data.csv',index=0)

train_data=data[~data.label.isnull()]
test_data=data[data.label.isnull()]
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


fea = [i for i in train_data.columns if i not in ['id','label']]

data = train_data[fea].copy()
target = train_data['label'].copy()
X_train,X_test,y_train,y_test = train_test_split(train_data[fea].values,train_data['label'],test_size=0.2,random_state=2020)

def custom_f1_eval(y_true, y_pred):
    y_pred_label = []
    for i in y_pred:
        if i<=0.5:
            y_pred_label.append(0)
        else:
            y_pred_label.append(1)
    
    f1_mean = f1_score(y_true, y_pred_label)    
    return "f1", f1_mean, True
 
def f1_loss(y, pred):
    beta = 2
    p = 1. / (1 + np.exp(-pred))
    grad = p * ((beta - 1) * y + 1) - beta * y
    hess = ((beta - 1) * y + 1) * p * (1.0 - p)
 
    return grad, hess

model=lgb.LGBMClassifier(
            n_estimators=100000,
            num_leaves=63,
            learning_rate=0.01,
            max_depth=6,
            metric=None,
            feature_fraction= 0.4,
            bagging_fraction=0.8,
            min_data_in_leaf= 16,
            is_unbalance=True
            )
model.set_params(**{"objective": f1_loss})
model.fit(X_train,
          y_train,
          eval_set=[(X_train,y_train),(X_test,y_test)],
          eval_metric=lambda y_true, y_pred: [custom_f1_eval(y_true, y_pred)],
          early_stopping_rounds=200,
          verbose=100,
         )
pred=model.predict(test_data[fea])

test_data['score'] = pred
sub = pd.read_csv('../data/entprise_evaluate.csv')
sub.drop(['score'],axis=1,inplace=True)
sub = sub.merge(test_data[['id','score']],on='id',how='left')
sub.to_csv('result.csv',index=0)
