import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, sys.path[0]+"/../")
from data.Ashare import *
# 导入数据
data=get_price('600519.XSHG',frequency='15m',count=100)  # 贵州茅台15m线实时行情，total 100' 
used_features = ["high", "low", "open", "volume"]
X = data[used_features]
y = data["close"]

# 从数据集中取30%作为测试集，其他作为训练集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型
regr.fit(X_train, y_train)

# 用训练得出的模型进行预测
diabetes_y_pred = regr.predict(X_test)


# 以 预测准确率=（预测正确样本数）/（总测试样本数）* 100% 对预测准确率进行计算，设定 ErrorTolerance = 5%
def accuracy(predict, true):
    sizeofall = len(true)
    sizeofright = 0
    for i in range(0, sizeofall):
        est = abs(predict[i] - true[i]) / true[i]
        if est < 0.05:
            sizeofright = sizeofright + 1

    return sizeofright/sizeofall


# 将预测准确率打印出来
predict = np.array(diabetes_y_pred)
true = np.array(y_test)
Ac = accuracy(predict, true)
print("Accuracy=", Ac*100, '%')

# 以 R-Squared 对预测准确率进行计算，将其打印出来
print("R-Squared Accuracy=", (regr.score(X_test, y_test)) * 100, '%')

# 将测试结果以图标的方式显示出来
plt.figure()
plt.plot(range(len(diabetes_y_pred)), diabetes_y_pred, 'go-', label="predict value")
plt.plot(range(len(diabetes_y_pred)), y_test, 'ro-', label="true value")
plt.legend()
plt.show()