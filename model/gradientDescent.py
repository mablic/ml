import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = 'D:\\python\\ML\\model\\data\\data1.txt'


class GradientDescent: 

    __yCol = pd.Series()
    __xCol = pd.DataFrame()
    __yColName = ''

    def __init__(self, data):
        self.df = data

    def setY(self, y_column):

        yList = []
        try:
            yList = self.df[y_column].tolist()
        except ValueError:
            print('%s is not valid.' % y_column)
        self.__yCol = pd.Series(yList)
        self.__yColName = y_column

    def setX(self, *args):
        columns = self.df.columns.tolist()
        xDict = {}
        nameSet = set()
        for i in range(len(args)):
            if args[i] not in nameSet:
                nameSet.add(args[i])

        for i in range(len(columns)):
            if columns[i] in nameSet and columns[i] not in xDict.keys():
                xDict[columns[i]] = self.df[columns[i]]
        ttl = len(self.df)
        xData = pd.DataFrame([1 for _ in range(ttl)], columns=['fst'])
        xData1 = pd.DataFrame(xDict)
        self.__xCol = xData.join(xData1)

    def computeCost(self, theta):
        if len(self.__yCol) == 0 or len(self.__xCol) == 0:
            return 0

        ttl = len(self.df)

        # return xData
        cost = 1 / (2 * ttl) * sum((np.array(self.__xCol).dot(theta) - self.__yCol) ** 2)
        return cost

    def computeGradientDescent(self, theta, alpha, nums):
        if len(self.__yCol) == 0 or len(self.__xCol) == 0:
            return 0

        ttl = len(self.df)
        ans = []

        for i in range(nums):
            theta = theta - alpha * (1/ttl) * ((np.transpose(np.array(self.__xCol).dot(theta) - self.__yCol).dot(np.array(self.__xCol))))
            ans.append(self.computeCost(theta))
        return theta, ans

    def graph(self, theta):
        fig, ax = plt.subplots(figsize=(6, 4))
        tmp = theta * self.__xCol
        self.df['predict'] = tmp.sum(axis=1)
        # plt.scatter(self.df.values)
        ax.plot(self.df['predict'], 'o', label='Prediction', color='g')
        ax.plot(self.df[self.__yColName], '^', label='Ground Truth', color='r')
        # ax.set_xlim((-1, 10))
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Prediction vs Actual')

        plt.legend(loc='upper right')
        plt.show()


def read_data():
    df = pd.read_csv(PATH, header=None)
    df.columns = ['x', 'y']
    return df


if __name__ == '__main__':

    df = read_data()
    # theta = [0, 0]
    # print(computeCost(df, theta))
    # predict = gradient_descent(df, theta, 0.01, 500)
    # print(predict)
    # graph_predict(df, theta)
    theta = np.array([0, 0])
    test = GradientDescent(df)
    test.setX('x')
    test.setY('y')
    ret = test.computeCost(theta)
    ret, ans = test.computeGradientDescent(theta, 0.01, 500)
    test.graph(ret)
    # print(ret)
