import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PATH = 'D:\\python\\ML\\model\\data\\data1.txt'


def read_data():
    df = pd.read_csv(PATH, header=None)
    df.columns = ['x', 'y']
    return df


def computeCost(data, t):
    m = len(data.y)
    cost = 1 / (2 * m) * sum((t[0] + t[1] * data.x - data.y) ** 2)
    return cost


def gradient_descent(data, t, alpha, nums):
    m = len(data.y)
    ans = []
    for i in range(nums):
        t0 = sum((t[0] + t[1] * data.x - data.y))
        t1 = sum((t[0] + t[1] * data.x - data.y) * df.x)

        t[0] -= (alpha/m) * t0
        t[1] -= (alpha/m) * t1
        ans.append(computeCost(data, t))

    return ans


def graph_predict(data, t):

    plt.scatter(data.x, data.y, c='r')
    plt.plot(data.x, t[0] + data.x * t[1], '-')
    plt.show()


if __name__ == '__main__':

    df = read_data()
    theta = [0, 0]
    predict = gradient_descent(df, theta, 0.01, 500)
    graph_predict(df, theta)


