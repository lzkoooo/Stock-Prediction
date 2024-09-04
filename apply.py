# -*- coding = utf-8 -*-
# @Time : 2024/9/3 下午9:45
# @Author : 李兆堃
# @File : apply.py
# @Software : PyCharm
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 加载最优模型
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

"""
目前LSTM预测非平稳变化序列任务时会出现滞后现象
主要是因为在数据急剧变化的时候，模型倾向于学习临近于预测天的数据（偷懒），这样可以使用最小的代价来获得最好的表现
对于这类的解决方法有
① 一阶差分（仅对序列趋势明显的数据有效），但对股票预测这类的非平稳变化序列任务仍无法解决，只能一定程度缓解
② 在时间序列回归问题中，不要直接给出希望模型预测的未经处理的真实值。
可以对输入样本进行非线性化的处理，平方，根号，ln等，是不能直接直观地预测其结果，而只是为算法提供模式。
尝试预测时间t和t-1处值的差异，而不是直接预测t时刻的值
③ 构造更加丰富的时序特征
存在问题：在后续的预测中，所使用的数据是来自于真实值还是预测值？
目前主流研究方向：将非平稳数据转为平稳数据进行输入，模型的输出也为平稳输出，再将此输出转为非平稳输出即可
"""

def Test_Data_Preprocessing(mem_his_days, pre_days):
    df = pd.read_csv('Data/Test_Stock_Data.csv', index_col=0)  # 读取数据
    df.dropna(inplace=True)  # 删除空值且直接替换（即在原数据上直接修改）
    df.sort_index(inplace=True)

    df['label'] = df['Close']
    Y_test = df['label'].values[(pre_days + mem_his_days - 1):]  # df['label']的格式是pandas.core.series.Series
    # 因为凭今天及之后mem的数据预测pre天后的值，那么构造训练集时今天数据的label就对应pre+mem天后的值
    # 因为下标从0开始，那么也就是下标(pre_days + mem_his_days - 1)对应第pre_days + mem_his_days个label

    scaler = joblib.load('scaler.pkl')
    sca_X = scaler.transform(df.iloc[:, :-1])  # 执行标准化，不包含label！！label不用标准化
    '''
        即loc[x, y]是指按值选取
        而iloc[x, y]是指按下标选取
        x为行，y为列
        :代表选择所有
        选取某一列可以用iloc[i]，选取完的也不是dataframe而是series
        选取某一行的数据，可以使用iloc[[i]]或者iloc[:, i]
    '''

    X_test = []
    # 构造时间数据滑动窗口
    deq = deque(maxlen=mem_his_days)
    for i in sca_X:
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X_test.append(list(deq))

    X_test = X_test[:-pre_days]
    X_test = np.array(X_test)  # y已经是array了
    return X_test, Y_test


def plot_result(Y_test, prediction):
    plt.plot(Y_test, label='True', color='blue')
    plt.plot(prediction, label='Predict', color='red')
    plt.title("Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    return plt

# 画图
if __name__ == '__main__':
    best_model = load_model('best_model')
    # best_model.summary()
    X_test, Y_test = Test_Data_Preprocessing(mem_his_days=5, pre_days=10)

    # results = best_model.evaluate(X_test, Y_test, verbose=1)

    # 输出每个指标的结果
    # for name, value in zip(best_model.metrics_names, results):
    #     print(f"{name}: {value}")
    prediction = best_model.predict(X_test)
    # print(prediction)
    prediction = [item for i in prediction for item in i]
    # print(prediction)

    # 画图
    plt = plot_result(Y_test, prediction[15:])
    plt.savefig('result.png')
    plt.show()
