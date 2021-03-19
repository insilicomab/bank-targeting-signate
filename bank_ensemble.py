# -*- coding: utf-8 -*-

'''
アンサンブル学習
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
from scipy import stats

# 予測データの読み込み
lgb_sub = pd.read_csv('./submit/bank_LightGBM.csv', sep=',', header=None)
xgb_sub = pd.read_csv('./submit/bank_xgboost.csv', sep=',', header=None)
rf_sub = pd.read_csv('./submit/bank_rf.csv', sep=',', header=None)

# 予測データの結合
df = pd.concat([lgb_sub[1], 
                xgb_sub[1],
                rf_sub[1]
                ],
               axis=1)

# アンサンブル学習（平均）
pred = df.mean(axis=1)

'''
提出
'''

# 提出用データの読み込み
sub = pd.read_csv('./data/submit_sample.csv', sep=',', header=None)
print(sub.head())
    
# 目的変数カラムの置き換え
sub[1] = pred

# ダミー変数をもとの変数に戻す
sub[1] = sub[1].replace([0,1,2], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# ファイルのエクスポート
sub.to_csv('./submit/bank_ensemble.csv', header=None, index=None)