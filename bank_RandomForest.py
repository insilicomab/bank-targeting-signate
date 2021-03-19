# -*- coding: utf-8 -*-

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
clipping
'''

# balanceのヒストグラム
plt.hist(df['balance'], bins=100)
plt.title('balance')
plt.show()

# 1%、99%点を計算し、clipping
p01 = df['balance'].quantile(0.01)
p99 = df['balance'].quantile(0.99)
df['balance'] = df['balance'].clip(p01, p99)

# clipping後のbalanceのヒストグラム
plt.hist(df['balance'], bins=100)
plt.title('balance')
plt.show()

'''
特徴量datetime作成
'''

# month を文字列から数値に変換
month_dict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, 
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
df["month_int"] = df["month"].map(month_dict)

# month と day を datetime に変換
data_datetime = df.assign(ymd_str=lambda x: "2014" + "-" + x["month_int"].astype(str) + "-" + x["day"].astype(str)).assign(datetime=lambda x: pd.to_datetime(x["ymd_str"]))["datetime"].values

# datetime を int に変換する
index = pd.DatetimeIndex(data_datetime)
df["datetime_int"] = np.log(index.astype(np.int64))

# 不要な列を削除
df = df.drop(["month", "day", "month_int"], axis=1)

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
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from statistics import mean

# 10分割する
folds = 10
skf = StratifiedKFold(n_splits=folds)

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
    
    model = rf(n_estimators=100,
               random_state=1234)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    auc = roc_auc_score(y_valid, y_pred)
    print(auc)
    
    models.append(model)
    aucs.append(auc)

# 平均AUCを計算する
print(mean(aucs))

"""
予測精度：
0.923898381400048
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
sub.to_csv('./submit/bank_baseline_rf.csv', header=None, index=False)

"""
スコア：
0.9358971
"""