import numpy as np
import matplotlib.pyplot as plt


def likelihood(x, y, beta):
    '''
    :param x: the data matrix
    :param y: the label matrix
    :param beta: the parameter vector in 3.27
    :return: the log-likelihood of 3.27
    '''
    sum = 0
    m, n = np.shape(x)
    for i in range(m):
        sum += (-y[i] * np.dot(beta, x[i].T) + np.log(1 + np.exp(np.dot(beta, x[i].T))))
    return sum

def loadDataSet(file_name):
    dataset = np.loadtxt(file_name, delimiter=",")
    dataMat = dataset[:, 1:3]
    labelMat = dataset[:, 3]
    return dataMat, labelMat


def plotBestFit(dataMat, labelMat):
    fig1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    print(dataMat[labelMat == 0, 0])
    plt.scatter(dataMat[labelMat == 0, 0], dataMat[labelMat == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(dataMat[labelMat == 1, 0], dataMat[labelMat == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("../data/watermelon_3a.csv")
    plotBestFit(dataMat, labelMat)
