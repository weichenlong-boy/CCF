# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:10:05 2020

@author: 月球来的火星人
"""
# =============================================================================
# catboost版本为0.24.1
# =============================================================================

import pandas as pd
import gc
import re 
import jieba
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import catboost as cbt
import xgboost as xgb
from tqdm import tqdm
import lightgbm as lgb
from datetime import datetime, timedelta
from scipy.special import boxcox1p
from scipy.stats import skew, norm
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox_normmax
import category_encoders as ce
from collections import Counter
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import  roc_auc_score,accuracy_score,f1_score
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
#import toad
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

# 读取数据
base_info = pd.read_csv('risk-prediction-of-illegal/base_info.csv')
tax_info = pd.read_csv('risk-prediction-of-illegal/tax_info.csv')
change_info = pd.read_csv('risk-prediction-of-illegal/change_info.csv')
other_info = pd.read_csv('risk-prediction-of-illegal/other_info.csv')
news_info = pd.read_csv('risk-prediction-of-illegal/news_info.csv')
entprise_info = pd.read_csv('risk-prediction-of-illegal/entprise_info.csv')
annual_report_info = pd.read_csv('risk-prediction-of-illegal/annual_report_info.csv')
test_evaluate = pd.read_csv('entprise_evaluate.csv')
test_submit = pd.read_csv('entprise_submit.csv')


feature = base_info.merge(entprise_info, how='left')

# =============================================================================
# 第二张表
# =============================================================================

for f in tqdm(['FUNDAM', 'EMPNUM', 'COLGRANUM','RETSOLNUM','DISPERNUM','UNENUM','COLEMPLNUM','RETEMPLNUM']):
    df_temp = annual_report_info.groupby('id')[f].agg(**{
        'annual_{}_mean'.format(f): 'mean',
        'annual_{}_std'.format(f): 'std',
        'annual_{}_max'.format(f): 'max',
        'annual_{}_min'.format(f): 'min',
        'annual_{}_sum'.format(f): 'sum',
        'annual_{}_count'.format(f): 'count', 
#         'annual_{}_skew'.format(f): 'skew',
    }).reset_index()
    feature = feature.merge(df_temp, how='left')

def unique_num(x):
    return len(np.unique(x))
for i in tqdm(['WEBSITSIGN', 'STATE']): 
    df_temp = annual_report_info.groupby('id',sort=False)[i].agg(**{'annual_{}_unique_num'.format(i): unique_num,}).reset_index()
    feature = feature.merge(df_temp, how='left')



# tax_info['START_DATE'] = pd.to_datetime(tax_info['START_DATE'])
# tax_info['END_DATE'] = pd.to_datetime(tax_info['END_DATE'])
# tax_info['DATE_gap'] = tax_info['END_DATE'].apply(lambda x: x.day) - tax_info['START_DATE'].apply(lambda x: x.day)
# def w2v_feat(df, group_id, feat, length):
#     print('start word2vec ...')
#     df[feat] = df[feat].astype('str')
#     data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
#     model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1,workers=1, hs=1, iter=10, seed=1)
#     data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
#     for m in range(length): 
#         data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
#     del data_frame[feat]
#     return data_frame
# feature = feature.merge(w2v_feat(change_info, 'id', 'bgxmdm', 5), how='left')
# def gen_user_group_amount_features(df, value):
#     group_df = df.pivot_table(index='id',
#                               columns=value,
#                               values='FUNDAM',
#                               dropna=False,
#                               aggfunc=['count','sum','mean']).fillna(0)
#     group_df.columns = ['id_{}_{}_FUNDAM_{}'.format(value, f[1], f[0]) for f in group_df.columns]
#     group_df.reset_index(inplace=True)
#     return group_df 
# # annual_report_info['WEBSITSIGN'] = annual_report_info['WEBSITSIGN'].fillna(-1)
# feature = feature.merge(gen_user_group_amount_features(df=annual_report_info, value='ANCHEYEAR'), how='left')
# def gen_user_countvec_features(df, value):
#     df[value] = df[value].astype(str)
#     df[value].fillna('-1', inplace=True)
#     group_df = df.groupby(['id']).apply(lambda x: x[value].tolist()).reset_index()
#     group_df.columns = ['id', 'list']
#     group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
#     enc_vec = CountVectorizer()
#     tfidf_vec = enc_vec.fit_transform(group_df['list'])
#     svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
#     vec_svd = svd_enc.fit_transform(tfidf_vec)
#     vec_svd = pd.DataFrame(vec_svd)
#     vec_svd.columns = ['svd_countvec_{}_{}'.format(value, i) for i in range(vec_svd.shape[1])]
#     group_df = pd.concat([group_df, vec_svd], axis=1)
#     del group_df['list']
#     return group_df
# feature = feature.merge(gen_user_countvec_features(df=annual_report_info, value='FUNDAM'), how='left')
# def w2v_feat(df, group_id, feat, length):
#     print('start word2vec ...')
#     df[feat] = df[feat].astype('str')
#     data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
#     model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1,workers=1, hs=1, iter=10, seed=1)
#     data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
#     for m in range(length): 
#         data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
#     del data_frame[feat]
#     return data_frame
# feature = feature.merge(w2v_feat(annual_report_info, 'id', 'EMPNUM', 5), how='left')
# groups = annual_report_info.groupby('id')
# indexs = annual_report_info['id'].unique()
# annual_fe = []
# for i in tqdm(['EMPNUMSIGN','STOCKTRANSIGN']): #,,'WEBSITSIGN','FORINVESTSIGN','STOCKTRANSIGN'
#     temp = [groups.get_group(g)[i].tolist()[-1] for g in groups.groups]
#     df_temp = pd.DataFrame({'id':indexs,f'{i}_state':temp})
#     annual_fe.append(f'{i}_state')
#     feature = feature.merge(df_temp, how='left')

# =============================================================================
# base
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
#     df['_date_gap'] = 4
#     df[df['date_gap']==30]['_date_gap'],df[df['date_gap']==40]['_date_gap'],df[df['date_gap']==50]['_date_gap'] = 1,2,3
    df['opfrom_year'] = df['opfrom'].apply(lambda x: x.year)
    df['opto_year'] = df['opto'].apply(lambda x: x.year)
    df['opfrom_month'] = df['opfrom'].apply(lambda x: x.month)
    df['opto_month'] = df['opto'].apply(lambda x: x.month)
    del df['opfrom'],df['opto']

    return df
feature = date_deal(feature)

# def f2(x):
#     if x==0.0:
#         return 1
#     elif x==1000.0:
#         return 2
#     else:
#         return 3
# feature['reccap_'] = feature['reccap'].apply(lambda x: f1(x))a
# feature[feature['label']==1][''opscope''].count()
# Counter(feature[feature['label']==1]['regcap']).most_common(50)
# Counter(feature[feature['label']==0]['regcap']).most_common(50)
# sns.distplot(feature[feature['label']==1]['regcap'], color="b", bins=10)
# sns.distplot(feature[feature['label']==0]['regcap'], color="b", bins=10)
# def w2v_feat(df, feat, length):
#     print('start word2vec ...')
#     data_frame = df.copy()
# #     df[feat] = df[feat].astype('str')
# #     data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
#     model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1,workers=1, hs=1, iter=10, seed=1)
#     data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
#     for m in range(length): 
#         data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
#     del data_frame[feat]
#     return data_frame
# df_ = w2v_feat(feature, 'opscope', 30)
# 读取停用词数据
# stopwords = pd.read_csv('../input/english-and-chinese-stopwords/cn_stopwords.txt', encoding='utf8', names=['stopword'], index_col=False)
# # 转化词列表
# stop_list = stopwords['stopword'].tolist()

# r =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：；“”‘’￥……（）《》【】]"
# seg = pkuseg.pkuseg() 
# feature['opscope'] = feature['opscope'].apply(lambda x: [i for i in seg.cut(re.sub(r,'',x)) if i not in stop_list])
# def f1(x):
#     if '投资' in x:
#         return 1
#     elif '咨询' in x:
#         return 2
#     elif '融资' in x:
#         return 3
#     else:
#         return 4
# feature['opscope_word'] = feature['opscope'].apply(lambda x: f1(x))
# def skew_deal(df,*cols):
#     for col in cols:
#         df[col] = df[col].apply(lambda x: np.log(1+x))
#     return df
# feature = skew_deal(feature,['empnum','regcap','exenum'])
# # 缺失值统计，统计存在缺失值的特征，构造缺失值相关计数特征
# loss_fea = ['enttypeitem', 'empnum', 'compform', 'venind', 'enttypeminu',
#        'reccap', 'opform_len', 'date_gap']
# for i in tqdm(loss_fea):
#     feature[i] = feature[i].fillna(-999)
#     a = feature.loc[feature[i]==-999]
#     e = a.groupby(['industryphy'])['id'].count().reset_index(name=i+'_industryphy_count') 
#     feature = feature.merge(e,on='industryphy',how='left')
    
#     d = a.groupby(['loanProduct'])['id'].count().reset_index(name=i+'_loan_count') 
#     data = data.merge(d,on='loanProduct',how='left')
    
#     m = a.groupby(['job'])['id'].count().reset_index(name=i+'_job_count') 
#     data = data.merge(m,on='job',how='left')
    
#     data['certloss_'+i] = data[i+'_certId_count']/data['certId_count']
#     data['jobloss_'+i] = data[i+'_job_count']/data['job_count']
#rank_fe = ['oplocdistrict','industryco','enttype','enttypeitem','enttypeminu']
# for i in tqdm(cross_fe):
#     feature[f'{i}_unique'] = feature.groupby([i])['id'].transform('nunique')
# dense_features = ['empnum','parnum','exenum','regcap','reccap']
# for f1 in tqdm(dense_features):
#     for f2 in dense_features:
#         if f1==f2:
#             continue
#         if '{}_add_{}'.format(f1,f2) not in feature.columns and '{}_add_{}'.format(f2,f1) not in feature.columns:
#             feature['{}_add_{}'.format(f1, f2)] = feature[f1] + feature[f2]
#             feature['{}_Mul_{}'.format(f1, f2)] = feature[f1] / feature[f2]
#feature['reg/rec'] = feature['regcap'] / feature['reccap']
#feature['emp/exe'] = feature['empnum'] / feature['exenum']
#feature['par/exe'] = feature['parnum'] / feature['exenum']
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
            
# =============================================================================
# CatBoostClassifier
# =============================================================================
ycol = 'label'
seed = 2020
num_folds=5
prediction = df_test[['id']]
prediction['score'] = 0
offline_score = []

for i,model_seed in enumerate(range(num_model_seed)):
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seeds[model_seed], shuffle=True).split(df_train[col], df_train[ycol])
    oof_probs = np.zeros(df_train.shape[0])
    for fold, (train_idx, valid_idx) in enumerate(kfold):
        X_train, y_train = df_train[col].iloc[train_idx], df_train[ycol].iloc[train_idx]
        X_valid, y_valid = df_train[col].iloc[valid_idx], df_train[ycol].iloc[valid_idx]

        model=CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            task_type="CPU",
            learning_rate=0.01,
            iterations=10000,
            random_seed=2020,
            od_type="Iter",
            depth=8,
            early_stopping_rounds=500)

        clf = model.fit(X_train,y_train, eval_set=(X_valid,y_valid),verbose=500)
        yy_pred_valid=clf.predict(X_valid)
        y_pred_valid = clf.predict(X_valid,prediction_type='Probability')[:,-1]
        oof_probs[valid_idx] = y_pred_valid
        offline_score.append(f1_score(y_valid, yy_pred_valid))
        pred_test = clf.predict(df_test[col],prediction_type='Probability')[:,-1]
        prediction['score'] += pred_test / num_folds
        
    print('OOF-MEAN-F1:%.6f, OOF-STD-F1:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    oof_probs += oof_probs / num_model_seed
    prediction['score'] += prediction['score'] / num_model_seed
