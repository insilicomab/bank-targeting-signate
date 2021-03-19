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
from sklearn.preprocessing import StandardScaler

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

# 説明変数と目的変数を指定
X_train = train.drop(['y', 'id'], axis=1)
Y_train = train['y']

# 説明変数の標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

'''
モデルの構築と評価
'''

# ライブラリのインポート
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.model_selection import cross_val_score

# K分割する
folds = 5
skf = StratifiedKFold(n_splits=folds)

# classifierのアルゴリズム全てを取得する 
allAlgorithms = all_estimators(type_filter="classifier")

# モデリングと正解率
allAlgorithms = all_estimators(type_filter="classifier")

for(name, algorithm) in allAlgorithms:
    
    if name == 'ClassifierChain':
        continue
    
    elif name == 'GaussianProcessClassifier':
        continue
    
    elif name == 'MultiOutputClassifier':
        continue
    
    elif name == 'OneVsOneClassifier':
        continue
    
    elif name == 'OneVsRestClassifier':
        continue
    
    elif name == 'OutputCodeClassifier':
        continue
    
    elif name == 'StackingClassifier':
        continue
    
    elif name == 'VotingClassifier':
        continue
    
    else:
        clf = algorithm()
        try:  # Errorがでるものがあるので、try文を入れる
            if hasattr(clf,"score"):
            # クロスバリデーション
                scores = cross_val_score(clf, X_train_std, Y_train, cv=skf)
                print(f"{name:<35}の正解率= {np.mean(scores)}")
    
        except:
            pass

"""
予測精度：

AdaBoostClassifier                 の正解率= 0.8992186179871279
BaggingClassifier                  の正解率= 0.8995502997175233
BernoulliNB                        の正解率= 0.8616192321999725
CalibratedClassifierCV             の正解率= 0.8916985668933162
CategoricalNB                      の正解率= nan
ComplementNB                       の正解率= nan
DecisionTreeClassifier             の正解率= 0.8770641101642374
DummyClassifier                    の正解率= 0.8829991184279141
ExtraTreeClassifier                の正解率= 0.8599603343519255
ExtraTreesClassifier               の正解率= 0.9006191455715017
GaussianNB                         の正解率= 0.8497867886486128
GradientBoostingClassifier         の正解率= 0.905263838048923
HistGradientBoostingClassifier     の正解率= 0.9076968411182886
KNeighborsClassifier               の正解率= 0.8902977539445679
LabelPropagation                   の正解率= 0.8698759038661776

"""

