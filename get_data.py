# -*- coding = utf-8 -*-
# @Time : 2024/9/4 下午12:43
# @Author : 李兆堃
# @File : get_data.py
# @Software : PyCharm
import pandas_datareader.data as web
import datetime

def save_data():
    start = datetime.datetime(2004, 1, 1)
    end = datetime.datetime(2021, 12, 31)
    df = web.DataReader('GOOGL', 'stooq', start, end)
    df.to_csv('Data/Ori_Stock_Data.csv')

    start = datetime.datetime(2022, 1, 1)
    end = datetime.datetime(2024, 9, 1)
    df = web.DataReader('GOOGL', 'stooq', start, end)
    df.to_csv('Data/Test_Stock_Data.csv')


if __name__ == '__main__':
    save_data()


