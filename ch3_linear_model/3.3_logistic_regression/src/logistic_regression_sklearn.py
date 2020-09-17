import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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
    plt.scatter(dataMat[labelMat == 0, 0], dataMat[labelMat == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(dataMat[labelMat == 1, 0], dataMat[labelMat == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("../data/watermelon_3a.csv")

    # generalization of test and train set
    data_train, data_test, label_train, label_test = model_selection.train_test_split(dataMat, labelMat, test_size=0.5, random_state=0)
    # model training
    log_model = LogisticRegression()
    log_model.fit(data_train, label_train)
    # model testing
    label_pre = log_model.predict(data_test)
    # summarize the accuracy of fitting
    print(metrics.confusion_matrix(label_test, label_pre))
    print(metrics.classification_report(label_test, label_pre))

    plotBestFit(dataMat, labelMat)