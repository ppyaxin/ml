import numpy as np
import pandas as pd


class LeastSquare:
    def __init__(self):
        self.intercept = 0
        self.coefficient = []
        self.r2 = 0
        self.rmse = 0

    def fit(self, x, y):
        # 复制x防止改变参数
        x = x[:]
        # 在x的第一列插入一列常数1
        x.insert(0, "constant", [1 for _ in range(len(x))])
        # 求x转置
        xt = x.T
        # (xt * x)
        head = np.dot(xt, x)
        # (xt * y)
        tail = np.dot(xt, pd.DataFrame(y))
        # beta = (xt * x) ^ -1 * (xt * y)
        beta = np.dot(np.linalg.inv(head), tail)
        # 将DataFrame转成list
        a = beta.T[0]
        # 截距
        self.intercept = a[0]
        # 斜率
        self.coefficient = a[1:]

        # 期望值（向量）
        expectations = np.dot(x, beta)
        # 实际值（向量）
        actual = list(y)
        # 平均值（标量）
        average = np.average(actual)

        # 期望值与实际值的平方差
        squire_errors = 0
        for i in range(len(expectations)):
            squire_errors += np.square(expectations[i][0] - actual[i])

        # 平均值与实际值的平方差
        actual_errors = 0
        for i in range(len(actual)):
            actual_errors += np.square(average - actual[i])

        # 计算R2
        self.r2 = 1 - squire_errors / actual_errors
        # 计算RMSE
        self.rmse = np.sqrt(squire_errors / len(actual))
