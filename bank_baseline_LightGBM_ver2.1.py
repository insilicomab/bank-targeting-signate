# -*- coding: utf-8 -*-

"""
コメント：
ver.2: 60歳以上かどうかの特徴量を追加。
ver.2.1: 層化サンプリング（StratifiedKFold）に変更
"""

'''
データの読み込みと確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# データの確認
train.head()
train.dtypes

'''
特徴量エンジニアリング
'''

# ライブラリのインポート
from sklearn.preprocessing import LabelEncoder

# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)

'''
特徴量（60歳以上か）を追加
'''

df['over60yr'] = df['age'].apply(lambda x:1 if x >= 60 else 0)

'''
LabelEncoderによるダミー変数化
'''

# object型の変数の取得
categories = df.columns[df.dtypes == 'object']
print(categories)

# 欠損値を数値に変換
for cat in categories:
    le = LabelEncoder()
    print(cat)
    
    df[cat].fillna('missing', inplace=True)
    le = le.fit(df[cat])
    df[cat] = le.transform(df[cat])
    # LabelEncoderは数値に変換するだけであるため、最後にastype('category')としておく
    df[cat] = df[cat].astype('category')

# trainとtestに再分割
train = df[~df['y'].isnull()]
test = df[df['y'].isnull()]

'''
モデルの構築と評価
'''

# ライブラリのインポート
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from statistics import mean

# 10分割する
folds = 10
skf = StratifiedKFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    'objective':'binary',
    'random_seed':1234    
}

# 説明変数と目的変数を指定
X_train = train.drop(['y', 'id'], axis=1)
Y_train = train['y']

# 各foldごとに作成したモデルごとの予測値を保存
models = []
aucs = []

for train_index, val_index in skf.split(X_train, Y_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)    
    
    model = lgb.train(params,
                      lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=100, # 学習回数の実行回数
                      early_stopping_rounds=20, # early_stoppingの判定基準
                      verbose_eval=10)
    
    y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_valid, y_pred)
    print(auc)
    
    models.append(model)
    aucs.append(auc)

# 平均AUCを計算する
print(mean(aucs))

# 特徴量重要度の表示
for model in models:
    lgb.plot_importance(model, importance_type='gain',
                        max_num_features=15)

"""
予測精度：
0.9333415592314921
"""

'''
テストデータの予測
'''

# テストデータの説明変数を指定
X_test = test.drop(['y', 'id'], axis=1)

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    preds.append(pred)

# predsの平均を計算
preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis = 0)


'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/submit_sample.csv', header=None)

# 'SalePrice'の値を置き換え
sub[1] = preds_mean

# CSVファイルの出力
sub.to_csv('./submit/bank_baseline_ver2.csv', header=None, index=False)

"""
スコア：
0.9401630
"""