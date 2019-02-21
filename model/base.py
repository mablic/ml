import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PATH = 'D:\\python\\ML\\model\\data\\data1.txt'


class Data:

    def __init__(self):
        self.theta = 0
        self.ans = []
        self.df_x = pd.DataFrame()
        self.df_y = pd.Series()

    def readData(self, *args, path=PATH):
        df = pd.read_csv(path, header=None)
        columns = []
        for i in args:
            columns.append(i)
        df.columns = columns
        return df

    def setY(self, y_column):

        yList = []
        try:
            yList = self.df[y_column].tolist()
        except ValueError:
            print('%s is not valid.' % y_column)
        return pd.Series(yList)
        # self.__yColName = y_column

    def setX(self, *args, valueParam=True):
        # valueParam = True means first column of the data is X0 data
        # it contains constant value
        # this feature is apply for linear regression

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
        xData = pd.DataFrame(xDict)
        if valueParam:
            tmpData = pd.DataFrame([1 for _ in range(ttl)], columns=['fst'])
            return tmpData.join(xData)
        else:
            return xData


class Graph:
    def __init__(self):
        pass

    def graph(self, theta, df, df_x, df_y):
        fig, ax = plt.subplots(figsize=(6, 4))
        tmp = theta * df_x
        df['predict'] = tmp.sum(axis=1)
        # plt.scatter(self.df.values)
        ax.plot(df['predict'], 'o', label='Prediction', color='g')
        ax.plot(df_y, '^', label='Ground Truth', color='r')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Prediction vs Actual')
        plt.legend(loc='upper right')
        plt.show()


# residual analysis
class Residual:
    def __init__(self):
        self.df = pd.DataFrame()

    def calcResidual(self, theta, df_x, df_y):
        # exit function for missing theta
        if len(theta) == 0 or len(df_x) == 0 or len(df_y) == 0:
            return
        self.df['residual'] = np.array(df_x).dot(theta) - df_y
        # least Square error analysis
        error = self.df['residual'].map(lambda x: x**2).sum()
        return error

    def plotResidual(self):
        plt.plot(self.df['residual'], 'o')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Residual Analysis')
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':

    TEST = Data()
    TEST.readData('x', 'y')
    TEST.setX('x')
    TEST.setY('y')

    # df = read_data('x', 'y')
    # theta = [0, 0]
    # print(computeCost(df, theta))
    # predict = gradient_descent(df, theta, 0.01, 500)
    # print(predict)
    # graph_predict(df, theta)
    # theta = np.array([0, 0])
    # test = GradientDescent(df)
    # test.setX('x')
    # test.setY('y')
    # ret = test.computeCost(theta)
    # ret, ans = test.computeGradientDescent(theta, 0.01, 500)
    # test.graph(ret)
    # print(ret)
