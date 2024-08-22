# -*- coding = utf-8 -*-
# @Time : 2024/8/19 下午11:16
# @Author : 李兆堃
# @File : preprocess.py
# @Software : PyCharm

"""
    数据预处理
"""

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
for i in sca_X:  # i的格式是numpy.ndarray
    deq.append(list(i))  # ！！！！！如果队列已经达到最大长度，此时从队尾添加数据，则队头的数据会自动出队！！！！！
    # print(deq)
    if len(deq) == mem_his_days:  # mem天为一个队列
        X.append(list(deq))  # 一个队列对应一个label，如deq0对应第15天的数据，即1~5天作为deq，预测10天后的数据，即预测第15天的数据
        # 由deq的特性可知，X的每一个deq实际上就是时间数据滑动窗口
        # X的范围是从1~-mem，因为第-mem+1个deq由于只有4个数据未加入X，即共有all - mem个deq

X_lately = X[-pre_days:]
# 剩下的pre天不用来记忆，仅用作前面mem数据的label，因为-pre-5天的label就是最后一天的数据
# 相对于每个deq的第一个下标i而言，该deq所预测的数据为i + 14天，那么当i为-pre-5天时，他所在的deq（-pre-5 ~ -pre）所预测的数据就是最后一天
X = X[:-pre_days]  # 真正用来mem的数据，1个下标 = 1个deq = 5天数据 = 1个label（10天后数据）

# print(X)
y = df['label'].values[mem_his_days - 1: -pre_days]  # df['label']的格式是pandas.core.series.Series
# 因label已shift 10，所以最后一天的数据实际是在label的-pre位置，而第一个deq的label则在df[label]的mem-1位置
# 好像不需要标准化？
'''
    后续可以考虑将shift 10 删除
'''

import numpy as np

X = np.array(X)  # y已经是array了

"""
    构建神经网络
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=X))
