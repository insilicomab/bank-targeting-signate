# -*- coding: utf-8 -*-

"""
コメント：
baseline作成後のEDA
"""

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

# 学習データの読み込み
train = pd.read_csv('./data/train.csv')

# 'id'の削除
train.drop('id', axis=1, inplace=True)

"""
コメント：
baseline作成後のEDA
"""

# ageのヒストグラムの描画
plt.hist(train['age'], bins=50)
plt.title('ageのヒストグラム')
plt.show()

# yの棒グラフの描画
y_counts = train['y'].value_counts()
y_counts.plot.bar(title='yの頻度')
plt.show()

"""
コメント：
baseline ver2.1作成後のEDA
"""

#　ライブラリのインポート
from sklearn.preprocessing import PowerTransformer

# durationのヒストグラム
plt.hist(train['duration'], bins=50)
plt.title('duration')
plt.show()

# Yeo-Johnson変換
pt = PowerTransformer(method='yeo-johnson')
data = train['duration'].values.reshape(-1,1)
pt.fit(data)
train['duration'] = pt.transform(data)

# Yeo-Johnson変換後のdurationのヒストグラム
plt.hist(train['duration'], bins=50)
plt.title('duration(Yeo-Johnson)')
plt.show()

# campaignのヒストグラム
plt.hist(train['campaign'], bins=50)
plt.title('campaign')
plt.show()

# Box-Cox変換
pt = PowerTransformer(method='box-cox')
data = train['campaign'].values.reshape(-1,1)
pt.fit(data)
train['campaign'] = pt.transform(data)

# Box-Cox変換後のcampaignのヒストグラム
plt.hist(train['campaign'], bins=50)
plt.title('campaign')
plt.show()

"""
コメント：
baseline ver.3作成後のEDA
（ver3でのスコア改善は見られず）
"""

# balanceのヒストグラム
plt.hist(train['balance'], bins=100)
plt.title('balance')
plt.show()

# 1%、99%点を計算し、clipping
p01 = train['balance'].quantile(0.01)
p99 = train['balance'].quantile(0.99)
train['balance'] = train['balance'].clip(p01, p99)

# clipping後のbalanceのヒストグラム
plt.hist(train['balance'], bins=100)
plt.title('balance')
plt.show()

"""
コメント：
baseline ver3.1作成後のEDA
"""

# balanceの対数化
train['balance_log'] = np.log(train['balance'] - train['balance'].min() + 1)
plt.hist(train['balance_log'], bins=100)
plt.title('balance_log')
plt.show()

# durationの対数化
train['duration_log'] = np.log(train['duration'] - train['duration'].min() + 1)
plt.hist(train['duration_log'], bins=100)
plt.title('duration_log')
plt.show()