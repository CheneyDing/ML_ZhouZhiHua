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


def p1(x, beta):
    return np.exp(np.dot(beta, x)) / (1 + np.exp(np.dot(beta, x)))

def dfunc(x, y, beta):
    '''
    :param x: the data matrix
    :param y: the label matrix
    :param beta: the parameter vector in 3.30
    :return: the derivative of the log-likelihood function
    '''
    m, n = np.shape(x)
    result = 0
    #print(x); print(y); print(beta)
    for i in range(m):
        result += x[i] * (y[i] - p1(x[i], beta))
    return -result.transpose()


def d2func(x, y, beta):
    '''
    :param x: the data matrix
    :param y: the label matrix
    :param beta: the parameter vector in 3.30
    :return: the second derivative of the log-likelihood function
    '''
    m, n = np.shape(x)
    d2f = np.zeros((m, m))
    for i in range(m):
        d2f[i][i] = p1(x[i], beta) * (1-p1(x[i], beta))
    result = np.mat(x.transpose()) * np.mat(d2f) * np.mat(x)
    return result


def newton(x, y, beta):
    '''
    :param x: the data matrix
    :param y: the label matrix
    :param beta: the initial parameter vector in 3.30
    :return: (the result parameter after training, the array of parameters during training)
    '''
    error = 0.0001
    max_times = 1000
    betaArr = []
    for i in range(max_times):
        df = dfunc(x, y, beta)
        if np.dot(df, df.transpose()) < error:
            break;
        d2f = d2func(x, y, beta)
        beta = beta - (df * np.linalg.inv(d2f)).A[0]
        betaArr.append(beta)
    return beta, betaArr


def gradDscent(x, y, beta):
    '''
    :param x: the data matrix
    :param y: the label matrix
    :param beta: the initial parameter vector in 3.30
    :return: (the result parameter after training, the array of parameters during training)
    '''
    m, n = np.shape(x)
    error = 0.00000001
    step = 0.1
    max_times = 10000
    betaArr = []
    for i in range(max_times):
        ll1 = likelihood(x, y, beta)
        beta = beta - step * dfunc(x, y, beta)
        betaArr.append(beta)
        ll2 = likelihood(x, y, beta)
        if abs(ll2 - ll1) < error:
            break
    return beta, betaArr


def loadDataSet(file_name):
    '''
    :param file_name: the path of data file
    :return: data matrix and label matrix
    '''
    dataset = np.loadtxt(file_name, delimiter=",")
    dataMat = dataset[:, 1:3]
    labelMat = dataset[:, 3]
    return dataMat, labelMat


def plotBestFit(dataMat, labelMat, beta, betaArr, index):
    '''
    :param dataMat: data matrix
    :param labelMat: label matrix
    :return: null
    '''
    fig1 = plt.figure(index)
    plt.subplot(2, 2, 1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    # print(dataMat[labelMat == 0, 0])
    plt.scatter(dataMat[labelMat == 0, 0], dataMat[labelMat == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(dataMat[labelMat == 1, 0], dataMat[labelMat == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='upper left')
    x = np.arange(0, 0.8, 0.01)
    y = [-(beta[2] + beta[0] * x[k]) / beta[1] for k in range(len(x))]
    plt.plot(x, y)
    # plot w1
    plt.subplot(2, 2, 2)
    plt.title('w1')
    plt.xlabel('train_times')
    plt.ylabel('value')
    x2 = np.arange(0, len(betaArr), 1)
    y2 = [betaArr[i][0] for i in range(len(x2))]
    plt.plot(x2, y2)
    # plot w2
    plt.subplot(2, 2, 3)
    plt.title('w2')
    plt.xlabel('train_times')
    plt.ylabel('value')
    x3 = np.arange(0, len(betaArr), 1)
    y3 = [betaArr[i][1] for i in range(len(x2))]
    plt.plot(x3, y3)
    # plot b
    plt.subplot(2, 2, 4)
    plt.title('b')
    plt.xlabel('train_times')
    plt.ylabel('value')
    x4 = np.arange(0, len(betaArr), 1)
    y4 = [betaArr[i][2] for i in range(len(x2))]
    plt.plot(x4, y4)
    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("../data/watermelon_3a.csv")
    # print(dataMat); print(labelMat)
    beta = [1, 1, 1]
    x = np.column_stack((dataMat[:, 0:2], np.ones(np.shape(dataMat)[0])))
    # gradient descent
    result, betaArr = gradDscent(x, labelMat, beta)
    print('Gradient Descent --- ' + 'train_times: ' + str(len(betaArr)) + '\tbeta: ' + str(result))
    result2, betaArr2 = newton(x, labelMat, beta)
    print('Gradient Descent --- ' + 'train_times: ' + str(len(betaArr2)) + '\tbeta: ' + str(result2))
    # plot result
    plotBestFit(dataMat, labelMat, result, betaArr, 1)
    plotBestFit(dataMat, labelMat, result2, betaArr2, 2)