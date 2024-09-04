# -*- coding = utf-8 -*-
# @Time : 2024/9/3 下午9:41
# @Author : 李兆堃
# @File : train.py
# @Software : PyCharm

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from preprocessing import Stock_Price_LSTM_Data_Preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def train_start(pre_days, the_mem_days, the_lstm_layers, the_dense_layers, the_units):
    X, Y = Stock_Price_LSTM_Data_Preprocessing(the_mem_days, pre_days)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)

    #filepath = r'D:\Git Hub Repositories\PostGraduate\Stock Prediction\Model\{:.2f}_{:02d}_mem_{}_lstm_{}_dense_{}_units_{}'.format(val_mape, epochs, the_mem_days, the_lstm_layers, the_dense_layers, the_units)
    # filepath = 'D:\Git Hub Repositories\PostGraduate\Stock Prediction\Weight\\' + '{val_mape:.2f}_{epoch:02d}' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units}'
    filepath = 'D:\Git Hub Repositories\PostGraduate\Stock Prediction\Model\\' + '{val_mape:.2f}_{epoch:02d}' + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units}'
    checkpoint = ModelCheckpoint(  # 回调函数，定期检查某指标并决定是否保存该模型（模型权重）
        filepath=filepath,
        save_weights_only=False,  # 如果为True，则只保存模型权重，否则保存整个模型
        monitor='val_mape',  # 监控的指标，即根据该指标决定是否保存模型，指定后则该指标变化时保存
        mode='min',  # monitor指标更小时保存
        save_best_only=True
    )

    """
        构建神经网络
    """
    model = Sequential()
    model.add(LSTM(the_units, input_shape=X.shape[1:], activation='relu', return_sequences=True))  # 为了调用cudnn加速激活函数改为tanh
    model.add(Dropout(0.1))

    # X.shape是(4276, 5, 5), 而[1:]则是取(5, 5)
    # LSTM中的input_shape一般为二维，前一参数为时间步，后一参数为特征数，即一个数据含5天，每天的一条有5个特征，即是一个5x5的矩阵
    # return_sequences=True表示返回整个序列，而不是最后一个输出，即返回每一个时间步的输出，而不是最后一个输出
    # LSTM层的输入是一个形状为(batch_size, timesteps, input_dim)的三维张量
    # units表示输出空间的维度，即输出层的神经元个数，也即输出数据的特征数

    # 第二层
    for i in range(the_lstm_layers):  # the_lstm_layers是几就加循环几次加几层
        model.add(LSTM(the_units, input_shape=X.shape[1:], activation='relu', return_sequences=True))  # 为了调用cudnn加速激活函数改为tanh
        model.add(Dropout(0.1))
    # 第三层
    model.add(LSTM(the_units, activation='relu'))
    model.add(Dropout(0.1))
    # 全连接层
    for i in range(the_dense_layers):
        model.add(Dense(the_units, activation='relu'))
        model.add(Dropout(0.1))
    # 输出层
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mape'])
    '''
        ## 编译模型
        loss通常用来衡量模型预测值与实际值之间的差距
        metrics用来评估模型性能
        MeanSquaredError即均方误差
        MeanAbsolutePercentageError即平均绝对百分比误差
    '''
    model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_test, Y_test), callbacks=[checkpoint], verbose=2)
    '''
        callbacks是回调函数，每次执行fit函数的时候都会自动调用
        在 Keras 中，callback 是一个类，它有多种形式，如 ModelCheckpoint、EarlyStopping、LearningRateScheduler等
    '''


if __name__ == '__main__':
    # 不同超参数寻找最优模型
    pre_days = 10
    mem_days = 5
    lstm_layers = [2, 3, 4]
    dense_layers = [2, 3, 4]
    units = [16, 32]
    # mem_days = [5]
    # lstm_layers = [5]
    # dense_layers = [2]
    # units = [16]


    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units in units:
                train_start(pre_days, mem_days, the_lstm_layers, the_dense_layers, the_units)
