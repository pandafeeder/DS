import numpy as np
import matplotlib.pyplot as plt
import operator

def createDateSet():
    group = np.array([[1., 1.1], [1., 1.], [0., 0.], [0, 0.1]])
    labels = np.array([['A'], ['A'], ['B'], ['B']])
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]][0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2Matrix(filename):
    with open(filename, 'r') as fd:
        count = len(fd.readlines())
    initMat = np.zeros([count, 3])
    labelList = []
    with open(filename, 'r') as fd:
        lineNum = 0
        for line in fd:
            line = line.strip()
            record = line.split('\t')
            initMat[lineNum] = record[:3]
            labelList.append(record[-1])
            lineNum += 1
    return initMat, np.array(labelList)




def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


if __name__ == '__main__':
    group, labels = createDateSet()
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [i[0] for i in group]
    y = [i[1] for i in group]
    ax.scatter(x, y)
    for i, l in enumerate(labels):
        ax.annotate(l[0], (x[i]-0.05, y[i]))
    data, label = file2Matrix('datingTestSet.txt')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(data[:,1], data[:,2])

    plt.show()
