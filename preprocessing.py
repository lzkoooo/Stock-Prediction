# -*- coding = utf-8 -*-
# @Time : 2024/8/19 下午11:16
# @Author : 李兆堃
# @File : preprocessing.py
# @Software : PyCharm

"""
    数据预处理
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import numpy as np
import joblib

"""
X 表示输入的特征数据
x 表示输入的特征数据中的一个样本
Y 表示整个数据集的标签或目标值
y 表示单个样本的标签或目标值
y^ 表示预测值
"""


def Stock_Price_LSTM_Data_Preprocessing(mem_his_days, pre_days):

    df = pd.read_csv('Data/Ori_Stock_Data.csv', index_col=0)  # 读取数据
    df.dropna(inplace=True)  # 删除空值且直接替换（即在原数据上直接修改）
    df.sort_index(inplace=True)

    df['label'] = df['Close']
    Y = df['label'].values[(pre_days + mem_his_days - 1):]  # df['label']的格式是pandas.core.series.Series
    # 因为凭今天及之后mem的数据预测pre天后的值，那么构造训练集时今天数据的label就对应pre+mem天后的值
    # 因为下标从0开始，那么也就是下标(pre_days + mem_his_days - 1)对应第pre_days + mem_his_days个label

    scaler = StandardScaler()
    sca_X = scaler.fit_transform(df.iloc[:, :-1])  # 执行标准化，不包含label！！label不用标准化
    joblib.dump(scaler, 'scaler.pkl')   # 保存scaler
    '''
        即loc[x, y]是指按值选取
        而iloc[x, y]是指按下标选取
        x为行，y为列
        :代表选择所有
        选取某一列可以用iloc[i]，选取完的也不是dataframe而是series
        选取某一行的数据，可以使用iloc[[i]]或者iloc[:, i]
    '''

    X = []
    # 构造时间数据滑动窗口
    deq = deque(maxlen=mem_his_days)
    for i in sca_X:  # i的格式是numpy.ndarray
        deq.append(list(i))  # ！！！！！如果队列已经达到最大长度，此时从队尾添加数据，则队头的数据会自动出队！！！！！
        # print(deq)
        if len(deq) == mem_his_days:  # mem天为一个队列
            X.append(list(deq))  # 一个队列对应一个label，如deq0对应第10天的close，即1~5天作为deq，预测10天后的数据，即预测第10天的数据
            # 由deq的特性可知，X的每一个deq实际上就是时间数据滑动窗口
            # X的范围是从1~-mem + 1，因为第-mem+2个deq由于只有9个数据未加入X，即共有all - mem + 1个deq

    X = X[:-pre_days]  # 真正用来mem的数据，每一条都对应一个label
    # 相当于第1天预测第20天，也就是19天后的数据，而deq加入过程中已经去掉9个，此处只需再去掉10个就可以
    # 因为deq范围为1~-mem，此时再选取1~-pre，则共选取1~-（pre+mem）

    X = np.array(X)  # y已经是array了
    return X, Y


if __name__ == '__main__':
    X, Y = Stock_Price_LSTM_Data_Preprocessing(10, 10)
    print(len(X))
    print(len(Y))
