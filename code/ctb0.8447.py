# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:02:45 2020

@author: 月球来的火星人
"""
# =============================================================================
# catboost版本为0.24.2
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import category_encoders as ce
import warnings
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

#import toad
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

# 读取数据
base_info = pd.read_csv("./train/base_info.csv")
annual_report_info = pd.read_csv("./train/annual_report_info.csv")
tax_info = pd.read_csv("./train/tax_info.csv")
change_info = pd.read_csv("./train/change_info.csv")
news_info = pd.read_csv("./train/news_info.csv")
other_info = pd.read_csv("./train/other_info.csv")
entprise_info = pd.read_csv("./train/entprise_info.csv")
test_evaluate = pd.read_csv("./entprise_evaluate.csv")
#test_submit = pd.read_csv('../input/3214132/entprise_submit.csv')
feature = base_info.merge(entprise_info, how='left')


# =============================================================================
# 第二张表
# =============================================================================
annual_report_info['school_num_all'] = annual_report_info['COLGRANUM'] + annual_report_info['COLEMPLNUM']
annual_report_info['ret_num_all'] =annual_report_info['RETSOLNUM'] + annual_report_info['RETEMPLNUM']
annual_report_info['dis_num_all'] = annual_report_info['DISPERNUM'] + annual_report_info['DISEMPLNUM']
annual_report_info['une_num_all'] = annual_report_info['UNENUM'] + annual_report_info['UNEEMPLNUM']
annual_report_info['employer_num'] = annual_report_info['COLGRANUM'] + annual_report_info['RETSOLNUM'] + annual_report_info['DISPERNUM'] + annual_report_info['UNENUM']
annual_report_info['employee_num'] = annual_report_info['COLEMPLNUM'] + annual_report_info['RETEMPLNUM'] + annual_report_info['DISEMPLNUM'] + annual_report_info['UNEEMPLNUM']

#feature = feature.merge(annual_report_info[['id','school_num_all']], how='left')
for f in tqdm(['FUNDAM', 'EMPNUM', 'COLGRANUM','RETSOLNUM','DISPERNUM','UNENUM','COLEMPLNUM','RETEMPLNUM']):
    df_temp = annual_report_info.groupby('id')[f].agg(**{
        'annual_{}_mean'.format(f): 'mean',
        'annual_{}_std'.format(f): 'std',
        'annual_{}_max'.format(f): 'max',
        'annual_{}_min'.format(f): 'min',
        'annual_{}_sum'.format(f): 'sum',
        'annual_{}_count'.format(f): 'count', 
    }).reset_index()
    feature = feature.merge(df_temp, how='left')
    
for f in tqdm(['school_num_all','ret_num_all','dis_num_all','une_num_all']):
    df_temp = annual_report_info.groupby('id')[f].agg(**{
        'annual_{}_mean'.format(f): 'mean',
    }).reset_index()
    feature = feature.merge(df_temp, how='left')

def unique_num(x):
    return len(np.unique(x))

#作两个特征的交叉
def cross_two(name_1,name_2):
    new_col=[]
    encode=0
    dic={}
    val_1=annual_report_info[name_1]
    val_2=annual_report_info[name_2]
    for i in tqdm(range(len(val_1))):
        tmp=str(val_1[i])+'_'+str(val_2[i])
        if tmp in dic:
            new_col.append(dic[tmp])
        else:
            dic[tmp]=encode
            new_col.append(encode)
            encode+=1
    return new_col

for i in tqdm(['WEBSITSIGN','STATE']): #, 'STATE'
    df_temp = annual_report_info.groupby('id',sort=False)[i].agg(**{'annual_{}_unique_num'.format(i): unique_num,}).reset_index()
    feature = feature.merge(df_temp, how='left')

new_col=cross_two('WEBSITSIGN', 'STATE')#作交叉特征
annual_report_info['WEBSITSIGN_STATE']=new_col
COL = ['id','WEBSITSIGN_STATE']#,'FORINVESTSIGN','ANCHEYEAR'  #,'PUBSTATE','STOCKTRANSIGN'
feature = pd.merge(feature,annual_report_info[COL],on=['id'],left_index=True,right_index=True,how='left')


# =============================================================================
# 处理base表
# =============================================================================
#删除缺失值99%以上的
del feature['midpreindcode'],feature['ptbusscope'],feature['protype'],feature['forregcap'],feature['forreccap']
feature['dom_len'] = feature['dom'].apply(lambda x: len(x))
feature['opscope_len'] = feature['opscope'].apply(lambda x: len(x))
feature['oploc_len'] = feature['oploc'].apply(lambda x: len(x))
feature['opform_len'] = feature['opform'].apply(lambda x: len(x) if type(x)==str else x)

def date_deal(df):
    df['opfrom'] = pd.to_datetime(df['opfrom'])
    df['opto'] = pd.to_datetime(df['opto'])
    df['date_gap'] = df['opto'].apply(lambda x: x.year) - df['opfrom'].apply(lambda x: x.year)
    df['opfrom_year'] = df['opfrom'].apply(lambda x: x.year)
    df['opto_year'] = df['opto'].apply(lambda x: x.year)
    df['opfrom_month'] = df['opfrom'].apply(lambda x: x.month)
    df['opto_month'] = df['opto'].apply(lambda x: x.month)
    del df['opfrom'],df['opto']
    return df
feature = date_deal(feature)

feature['opform'] = feature['opform'].replace('01', '01-以个人财产出资').replace('02', '02-以家庭共有财产作为个人出资')
feature['opform_'] = feature['opform'].fillna(-1)

def f(x):
    if x=='10':
        return 1
    elif x=='01-以个人财产出资':
        return 2
    elif x==-1:
        return 3
    else:
        return 4
feature['opform_'] = feature['opform_'].apply(lambda x: f(x))

#筛选类别特征
feature_list = ['oplocdistrict','industryphy','enttype','state','adbusign','townsign',
                'regtype','opform','enttypegb','dom_len','oploc_len'] #'opscope_len'

cross_fe = []
for c in tqdm(feature_list):
    for c1 in ['industryphy']:
        if c1==c:
            continue
        if 'cross_{}_{}'.format(c,c1) not in feature.columns and 'cross_{}_{}'.format(c1,c) not in feature.columns:
            feature['cross_{}_{}'.format(c,c1)]=feature[c].astype(str)+feature[c1].astype(str)
            cross_fe.append('cross_{}_{}'.format(c,c1))

for i in tqdm(feature_list):
    feature[f'{i}_count'] = feature.groupby([i])['id'].transform('count')
    
TE_encoder = ce.TargetEncoder(cols = feature_list)
feature = TE_encoder.fit_transform(feature, feature['label'])

pre_col = ['id','opscope','dom','label','oploc']
cat_fe = [i for i in feature.select_dtypes('object').columns if i not in pre_col]
for f in tqdm(cat_fe):
    lbl = LabelEncoder()
    feature[f] = lbl.fit_transform(feature[f].astype(str))

df_train = feature[~feature['label'].isnull()].copy().reset_index(drop=True)
target = df_train['label']
df_test = feature[feature['label'].isnull()].copy().reset_index(drop=True)

col = [i for i in feature.columns if i not in pre_col]
df_train.to_csv('./train_features.csv',index=0)
df_test.to_csv('./test_features.csv',index=0)

# =============================================================================
# CatBoostClassifier
# =============================================================================
ycol = 'label'
seed = 2020
num_folds=5
kfold = StratifiedKFold(n_splits=num_folds, random_state=2020, shuffle=False).split(df_train[col], df_train[ycol])

prediction = df_test[['id']]
prediction['score'] = 0
oof_probs = np.zeros(df_train.shape[0])
offline_score = []
feature_importance_df = pd.DataFrame()

for fold, (train_idx, valid_idx) in enumerate(kfold):
    X_train, y_train = df_train[col].iloc[train_idx], df_train[ycol].iloc[train_idx]
    X_valid, y_valid = df_train[col].iloc[valid_idx], df_train[ycol].iloc[valid_idx]

    model=CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="F1",
        task_type="CPU",
        learning_rate=0.02,
        iterations=10000,
        random_seed=2020,
        od_type="Iter",
        l2_leaf_reg=3,
        depth=8,
        early_stopping_rounds=1000,
    )

    clf = model.fit(X_train,y_train, eval_set=(X_valid,y_valid),verbose=500)
    yy_pred_valid=clf.predict(X_valid)
    y_pred_valid = clf.predict(X_valid,prediction_type='Probability')[:,-1]
    oof_probs[valid_idx] = y_pred_valid
    offline_score.append(f1_score(y_valid, yy_pred_valid))
    pred_test = clf.predict(df_test[col],prediction_type='Probability')[:,-1]
    prediction['score'] += pred_test / num_folds
    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = model.feature_names_
    fold_importance_df["importance"] = model.feature_importances_
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print('OOF-MEAN-F1:%.6f, OOF-STD-F1:%.6f' % (np.mean(offline_score), np.std(offline_score)))
print('feature importance:')
feature_importance_df_ = feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False)
print(feature_importance_df_)
feature_importance_df_.to_csv("./importance.csv")
