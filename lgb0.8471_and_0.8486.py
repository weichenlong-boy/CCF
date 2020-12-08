# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:29:13 2020

@author: 月球来的火星人
"""
# =============================================================================
# lightgbm版本2.3.1
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from tqdm import tqdm
import xgboost as xgb
import catboost as cab
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_recall_fscore_support,roc_curve,auc,roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import category_encoders as ce
import lightgbm as lgb

PATH = '/risk-prediction-of-illegal/'
# 读取数据
base_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/base_info.csv')
tax_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/tax_info.csv')
change_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/change_info.csv')
other_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/other_info.csv')
news_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/news_info.csv')
entprise_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/entprise_info.csv')
annual_report_info = pd.read_csv('/kaggle/input/risk-prediction-of-illegal/annual_report_info.csv')

test_evaluate = pd.read_csv('../input/3214132/entprise_evaluate.csv')
test_submit = pd.read_csv('../input/3214132/entprise_submit.csv')

def filter_col_by_nan(df, ratio=0.05):
    cols = []
    for col in df.columns:
        if df[col].isna().mean() >= (1-ratio):
            cols.append(col)
    return cols

def unique_num(x):
    return len(np.unique(x))

# 剔除纯空列
base_info = base_info.drop(filter_col_by_nan(base_info, 0.01), axis=1)
annual_report_info = pd.read_csv(PATH + 'annual_report_info.csv')
annual_report_info = annual_report_info.drop(filter_col_by_nan(annual_report_info, 0.01), axis=1)
other_info = pd.read_csv(PATH + 'other_info.csv')
other_info = other_info[~other_info['id'].duplicated()]
other_info['other_SUM'] = other_info[['legal_judgment_num', 'brand_num', 'patent_num']].sum(1)
other_info['other_NULL_SUM'] = other_info[['legal_judgment_num', 'brand_num', 'patent_num']].isnull().astype(int).sum(1)

news_info = pd.read_csv(PATH + 'news_info.csv')
news_info['public_date'] = news_info['public_date'].apply(lambda x: x if '-' in str(x) else np.nan)
news_info['public_date'] = pd.to_datetime(news_info['public_date'])
news_info['public_date'] = (datetime.now() - news_info['public_date']).dt.days
news_info_df = news_info.groupby('id').agg({'public_date': ['count','max','min','mean']}).reset_index()
news_info_df.columns = ['id', 'public_date_COUNT', 'public_MAX', 'public_MIN', 'public_MEAN']
news_info_df2 = pd.pivot_table(news_info, index='id', columns='positive_negtive', aggfunc='count').reset_index()
news_info_df2.columns = ['id', 'news_COUNT1', 'news_COUNT2', 'news_COUNT3']
news_info_df = pd.merge(news_info_df, news_info_df2)

tax_info = pd.read_csv(PATH + 'tax_info.csv')
tax_info_df = tax_info.groupby('id').agg({
    'TAX_CATEGORIES': ['count'],
    'TAX_ITEMS': ['count'],
    'TAXATION_BASIS': ['count'],
    'TAX_AMOUNT': ['max', 'min', 'mean'],
})
tax_info_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() 
                                for e in tax_info_df.columns.tolist()])
tax_info_df = tax_info_df.reset_index()

change_info = pd.read_csv(PATH + 'change_info.csv')
change_info['bgrq'] = (change_info['bgrq'] / 10000000000).astype(int)

change_info_df = change_info.groupby('id').agg({
    'bgxmdm': ['count', 'nunique'],
    'bgq':['nunique'],
    'bgh': ['nunique'],
    'bgrq': ['nunique'],
})
change_info_df.columns = pd.Index(['changeinfo_' + e[0] + "_" + e[1].upper() 
                                for e in change_info_df.columns.tolist()])
change_info_df = change_info_df.reset_index()

annual_report_info = pd.read_csv(PATH + 'annual_report_info.csv')
annual_report_info_df = annual_report_info.groupby('id').agg({
    'ANCHEYEAR': ['max'],
    'STATE': ['max'],
    'FUNDAM': ['max'],
    'EMPNUM': ['max'],
    'UNEEMPLNUM': ['max', 'sum','mean']
})
annual_report_info_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() 
                                for e in annual_report_info_df.columns.tolist()])

annual_report_info_df = annual_report_info_df.reset_index()

base_info['district_FLAG1'] = (base_info['orgid'].fillna('').apply(lambda x: str(x)[:6]) == \
    base_info['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
base_info['district_FLAG2'] = (base_info['orgid'].fillna('').apply(lambda x: str(x)[:6]) == \
    base_info['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
base_info['district_FLAG3'] = (base_info['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6]) == \
    base_info['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)

base_info['person_SUM'] = base_info[['empnum', 'parnum', 'exenum']].sum(1)
base_info['person_NULL_SUM'] = base_info[['empnum', 'parnum', 'exenum']].isnull().astype(int).sum(1)

base_info['dom_len'] = base_info['dom'].fillna('暂无').apply(lambda x: len(x))
base_info['opscope_len'] = base_info['opscope'].fillna('暂无').apply(lambda x: len(x))
base_info['oploc_len'] = base_info['oploc'].fillna('暂无').apply(lambda x: len(x))
    
base_info['opfrom'] = pd.to_datetime(base_info['opfrom'])
base_info['opto'] = pd.to_datetime(base_info['opto'])

base_info['opfrom_TONOW'] = (max(base_info['opfrom']) - base_info['opfrom']).dt.days
base_info['opfrom_TIME'] = (base_info['opto'] - base_info['opfrom']).dt.days

base_info['opscope_COUNT'] = base_info['opscope'].apply(lambda x: len(x.replace("\t", "，").replace("\n", "，").split('、')))

cat_col = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
           'enttypeitem', 'enttypeminu', 'enttypegb',
          'dom', 'oploc', 'opform','opscope_len','oploc_len']

for col in cat_col:
    
    base_info[col + '_COUNT'] = base_info[col].map(base_info[col].fillna('暂无').value_counts())
    col_idx = base_info[col].value_counts()
    for idx in col_idx[col_idx < 10].index:
        base_info[col] = base_info[col].replace(idx, -1)
base_info = base_info.drop(['opfrom', 'opto'], axis=1)

for col in ['industryphy', 'dom', 'opform', 'oploc']:
    base_info[col] = pd.factorize(base_info[col])[0]

train_data = pd.merge(base_info, entprise_info, on='id')
train_data = pd.merge(train_data, other_info, on='id', how='left')

train_data = pd.merge(train_data, news_info_df, on='id', how='left')
train_data = pd.merge(train_data, tax_info_df, on='id', how='left')
train_data = pd.merge(train_data, annual_report_info_df, on='id', how='left')
train_data = pd.merge(train_data, change_info_df, on='id', how='left')

entprise_evaluate = test_evaluate[['id']]
test_data = pd.merge(base_info, entprise_evaluate, on='id')
test_data = pd.merge(test_data, other_info, on='id', how='left')
test_data = pd.merge(test_data, news_info_df, on='id', how='left')
test_data = pd.merge(test_data, tax_info_df, on='id', how='left')
test_data = pd.merge(test_data, annual_report_info_df, on='id', how='left')
test_data = pd.merge(test_data, change_info_df, on='id', how='left')

def eval_score(y_test,y_pre):
    _,_,f_class,_=precision_recall_fscore_support(y_true=y_test,y_pred=y_pre,labels=[0,1],average=None)
    fper_class={'合法':f_class[0],'违法':f_class[1],'f1':f1_score(y_test,y_pre)}
    return fper_class


def k_fold_serachParmaters(model,train_val_data,train_val_kind, test_kind):
    mean_f1=0
    mean_f1Train=0
    n_splits=5
    
    cat_features = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
           'enttypeitem', 'enttypeminu', 'enttypegb',
          'dom', 'oploc', 'opform']
    
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    pred_Test = np.zeros(len(test_kind))
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train, 
                  eval_set=[(x_test, y_test)], 
                  categorical_feature = cat_features,
                 early_stopping_rounds=100,
                 verbose=False)
        
        pred = model.predict(x_test)
        fper_class = eval_score(y_test,pred)
        
        pred_Train = model.predict(x_train)
        pred_Test += model.predict_proba(test_kind)[:, 1]/n_splits
        fper_class_train = eval_score(y_train,pred_Train)

        mean_f1 += fper_class['f1']/n_splits
        mean_f1Train+=fper_class_train['f1']/n_splits
        # print(mean_f1, mean_f1Train)
        
        
    return mean_f1, pred_Test

params0_ = [
dict(learning_rate=0.032, max_depth=12, min_child_samples=3,n_estimators=150, num_leaves=22, random_state=2000),
dict(learning_rate=0.131, max_depth=9, min_child_samples=6,n_estimators=150, num_leaves=22, random_state=2008),
dict(learning_rate=0.157, max_depth=8, min_child_samples=4,n_estimators=150, num_leaves=18, random_state=520),
dict(learning_rate=0.143, max_depth=8, min_child_samples=4,n_estimators=150, num_leaves=18, random_state=1949),
dict(learning_rate=0.149, max_depth=8, min_child_samples=4,n_estimators=150, num_leaves=18, random_state=99)
]
#8471
params1_ = [
dict(learning_rate=0.093, max_depth=9, min_child_samples=4,n_estimators=150, num_leaves=19, random_state=93),
dict(learning_rate=0.191, max_depth=9, min_child_samples=4,n_estimators=150, num_leaves=20, random_state=29), 
dict(learning_rate=0.139, max_depth=12, min_child_samples=4,n_estimators=150, num_leaves=13, random_state=67),
dict(learning_rate=0.149, max_depth=8, min_child_samples=4,n_estimators=150, num_leaves=18, random_state=99),
dict(learning_rate=0.02, max_depth=9, min_child_samples=4,n_estimators=225, num_leaves=16, random_state=268435456),
dict(learning_rate=0.061, max_depth=8, min_child_samples=6,n_estimators=322, num_leaves=21, random_state=16777216),
dict(learning_rate=0.038, max_depth=12, min_child_samples=4,n_estimators=317, num_leaves=20, random_state=262144),
dict(learning_rate=0.099, max_depth=10, min_child_samples=4,n_estimators=162, num_leaves=20, random_state=16384),
dict(learning_rate=0.191, max_depth=9, min_child_samples=4,n_estimators=150, num_leaves=20, random_state=29),
dict(learning_rate=0.116, max_depth=4, min_child_samples=8,n_estimators=150, num_leaves=13, random_state=39),
dict(learning_rate=0.137, max_depth=5, min_child_samples=9,n_estimators=150, num_leaves=23, random_state=30),
dict(learning_rate=0.131, max_depth=10, min_child_samples=3,n_estimators=150, num_leaves=18, random_state=45),
dict(learning_rate=0.15, max_depth=10, min_child_samples=6,n_estimators=150, num_leaves=18, random_state=33),
dict(learning_rate=0.079, max_depth=12, min_child_samples=4,n_estimators=150, num_leaves=21, random_state=88),
dict(learning_rate=0.147, max_depth=4, min_child_samples=9,n_estimators=150, num_leaves=24, random_state=56),
dict(learning_rate=0.15, max_depth=11, min_child_samples=2,n_estimators=150, num_leaves=14, random_state=19),
dict(learning_rate=0.147, max_depth=4, min_child_samples=9,n_estimators=150, num_leaves=23, random_state=40),
dict(learning_rate=0.154, max_depth=8, min_child_samples=8,n_estimators=150, num_leaves=17, random_state=41),
dict(learning_rate=0.033, max_depth=4, min_child_samples=3,n_estimators=150, num_leaves=12, random_state=46),
dict(learning_rate=0.15, max_depth=11, min_child_samples=2,n_estimators=150, num_leaves=14, random_state=19)]
#0.8486

score_tta = None
score_list = []
params = []
tta_fold = len(params1_)
for _ in tqdm(range(tta_fold)):
    clf = lgb.LGBMClassifier(**params1_[_])
    score, test_pred = k_fold_serachParmaters(clf,
                           train_data.drop(['id', 'opscope','label'], axis=1),
                           train_data['label'],
                           test_data.drop(['id', 'opscope'], axis=1)
                          )
    params.append(clf.get_params)
    if score_tta is None:
        score_tta = test_pred/tta_fold
    else:
        score_tta += test_pred/tta_fold
    score_list.append(score)
    
print(np.array(score_list).mean(), np.array(score_list).std())
dict_ = {i:j for i,j in zip(score_list,params)}
result = sorted(dict_.items(),key=lambda x:x[0],reverse=True)
result = test_data[['id']]
result['score'] = score_tta
result.to_csv('result.csv', index=None)
