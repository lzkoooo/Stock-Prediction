# -*- coding = utf-8 -*-
# @Time : 2024/8/19 下午11:16
# @Author : 李兆堃
# @File : preprocess.py
# @Software : PyCharm

import pandas_datareader.data as web
import datetime
from sklearn.preprocessing import StandardScaler
from collections import deque

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2021, 9, 1)
df = web.DataReader('GOOGL', 'stooq', start, end)

df.sort_index(inplace=True)
pre_days = 10
df['label'] = df['Close'].shift(-pre_days)  # 因为凭今天及之前数据预测10天后的值，那么构造训练集时今天数据的label就对应10天后的值
# print(df)


scaler = StandardScaler()
sca_X = scaler.fit_transform(df.iloc[:, :-1])  # iloc[:, :-1]表示取除了label列之外的所有列
print(sca_X)
'''
    即loc[x, y]是指按值选取，x为行，y为列
    而iloc[x, y]是指按下标选取，x为行，y为列
    选取某一列可以用iloc[i]，选取完的也不是dataframe而是series
    选取某一行的数据，可以使用iloc[[i]]
'''
# print(sca_X)

mem_his_days = 5  # 记忆天数，即通过mem天的数据去预测pre天后的数据

deq = deque(maxlen=mem_his_days)

X = []
for i in sca_X:
    deq.append(i)
    print(deq)
    if len(deq) == mem_his_days:
        X.append(list(deq))
X_lately = X[-pre_days:]    # 剩下的pre天不用来记忆，仅用作前面mem数据的label
X = X[:-pre_days]   # 真正用来mem的数据
print(X)
