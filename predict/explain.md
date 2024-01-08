# 导入数据
data=get_price('600519.XSHG',frequency='15m',count=100)  # 贵州茅台15m线实时行情，total 100' 
used_features = ["High Price", "Low Price", "Open Price", "Volume"]
X = data[used_features]
y = data["Close Price"]

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
plt.show() 以md形式解释他

GitHub Copilot: 这段代码主要用于使用线性回归模型对股票价格进行预测，并计算预测的准确率。

以下是代码的详细解释：

1. 导入所需的库：

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, sys.path[0]+"/../")
from data.Ashare import *
```

2. 获取股票数据，并选择需要的特征：

```python
data=get_price('600519.XSHG',frequency='15m',count=100)  # 贵州茅台15m线实时行情，total 100' 
used_features = ["High Price", "Low Price", "Open Price", "Volume"]
X = data[used_features]
y = data["Close Price"]
```

3. 将数据集分为训练集和测试集：

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)
```

4. 创建线性回归模型，并使用训练集进行训练：

```python
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
```

5. 使用训练得到的模型对测试集进行预测：

```python
diabetes_y_pred = regr.predict(X_test)
```

6. 定义一个函数来计算预测的准确率：

```python
def accuracy(predict, true):
    sizeofall = len(true)
    sizeofright = 0
    for i in range(0, sizeofall):
        est = abs(predict[i] - true[i]) / true[i]
        if est < 0.05:
            sizeofright = sizeofright + 1

    return sizeofright/sizeofall
```

7. 计算并打印预测的准确率：

```python
predict = np.array(diabetes_y_pred)
true = np.array(y_test)
Ac = accuracy(predict, true)
print("Accuracy=", Ac*100, '%')
```

8. 使用R-Squared计算预测的准确率，并打印：

```python
print("R-Squared Accuracy=", (regr.score(X_test, y_test)) * 100, '%')
```

9. 将预测结果和真实结果绘制在图表上：

```python
plt.figure()
plt.plot(range(len(diabetes_y_pred)), diabetes_y_pred, 'go-', label="predict value")
plt.plot(range(len(diabetes_y_pred)), y_test, 'ro-', label="true value")
plt.legend()
plt.show()
```

这段代码的主要目的是使用线性回归模型对股票价格进行预测，并计算预测的准确率。