# -*- coding = utf-8 -*-
# @Time : 2024/9/3 下午9:45
# @Author : 李兆堃
# @File : apply.py
# @Software : PyCharm

# 加载最优模型
from tensorflow.keras.models import load_model
best_model = load_model('')
best_model.summary()

best_model.evaluate(X_test, y_test)

# 画图
