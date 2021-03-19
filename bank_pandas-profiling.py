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

# pandas-profiling
profile = ProfileReport(train)
profile.to_file('profile_report.html')

