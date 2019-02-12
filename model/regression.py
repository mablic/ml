import pandas as pd
import numpy as np
import math
import model.base as base
import matplotlib.pyplot as plt

PATH1 = 'D:\\python\\ML\\model\\data\\data1.txt'
PATH2 = 'D:\\python\\ML\\model\\data\\log-data1.txt'


# this module contains regression class
# gradientdescent, logistic class
# this is the gradientdescent for linear regression
class GradientDescent(base.Data, base.Graph):

    def __init__(self, *args, path):
        self.theta = 0
        self.ans = []
        self.df_x = pd.DataFrame()
        self.df_y = pd.Series()
        self.df = super().readData(*args, path=path)

    def setX(self, *args):
        self.df_x = super().setX(*args, valueParam=True)

    def setY(self, y_column):
        self.df_y = super().setY(y_column)

    def computeCost(self, theta):

        if len(self.df_y) == 0 or len(self.df_x) == 0:
            return 0

        ttl = len(self.df)

        # return xData
        cost = 1 / (2 * ttl) * sum((np.array(self.df_x).dot(np.array(theta)) - self.df_y) ** 2)
        return cost

    def computeGradientDescent(self, theta, alpha, nums):
        if len(self.df_y) == 0 or len(self.df_x) == 0:
            return 0
        ttl = len(self.df)
        ans = []

        for i in range(nums):
            theta = theta - alpha * (1/ttl) * \
                    (np.transpose(np.array(self.df_x).dot(theta) - self.df_y).dot(np.array(self.df_x)))
            self.ans.append(self.computeCost(theta))
        self.theta = theta

    def graph(self):
        super().graph(self.theta, self.df, self.df_x, self.df_y)

    def graphRegression(self, xCol, theta):

        plt.plot(self.df_x[xCol], self.df_y, "o")
        plt.plot(self.df_x[xCol], self.df_x.dot(theta))
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Regression prediction')
        plt.legend(loc='upper right')
        plt.show()

class LogisticRegression(base.Data, base.Graph):

    def __init__(self, *args, path):
        self.theta = []
        self.ans = []
        self.df_x = pd.DataFrame()
        self.df_y = pd.Series()
        self.df = super().readData(*args, path=path)

    def setX(self, *args):
        self.df_x = super().setX(*args, valueParam=True)

    def setY(self, y_column):
        self.df_y = super().setY(y_column)

    def sigmoidFunction(self, z):
        return 1 / (1 + math.exp(-z))

    def computeCost(self, theta):
        # this is for the cost computation for logistic regression
        # it apply different cost function compare to linear regression
        if len(self.df_y) == 0 or len(self.df_x) == 0:
            return 0

        ttl = len(self.df)
        z = self.df_x.dot(np.array(theta))
        sigmoid = z.apply(self.sigmoidFunction)
        # compute Cost
        cost = (1 / ttl)*sum(-self.df_y * np.log(sigmoid)-(1-self.df_y)* np.log(1 - sigmoid))
        return cost

    def computeGradientDescent(self, theta, alpha, nums):
        if len(self.df_y) == 0 or len(self.df_x) == 0:
            return 0
        ttl = len(self.df)
        # ans = []
        theta = np.array(theta)
        for i in range(nums):
            z = self.df_x.dot(theta)
            sigmoid = z.apply(self.sigmoidFunction)
            theta = theta - alpha * (1/ttl) * (np.transpose(sigmoid - self.df_y)).dot(np.array(self.df_x))
            self.ans.append(self.computeCost(theta))
        self.theta = theta


if __name__ == '__main__':
    testObj = GradientDescent('x', 'y', path=PATH1)
    testObj.setX('x')
    testObj.setY('y')
    theta = [0, 0]
    testObj.computeGradientDescent(theta, 0.01, 500)
    testObj.graphRegression('x', testObj.theta)
    # print(testObj.theta)
    # print(testObj.computeCost(testObj.theta))
    # testObj = LogisticRegression('x1', 'x2', 'y', path=PATH2)
    # testObj.setX('x1', 'x2')
    # testObj.setY('y')
    # theta = [0, 0, 0]
    # testObj.computeGradientDescent(theta, 0.005, 100000)
    #
    # print(testObj.theta)
    # print(testObj.ans[-1])
    # print(testObj.ans)
    # print(testObj.computeCost(theta))


