import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sns.set()
plt.ion()

"""
    Reading the Dateset
"""


def Read(dirPath):
    df_train = pd.read_csv(os.path.join(dirPath, 'K_means_train.csv'))
    df_valid = pd.read_csv(os.path.join(dirPath, 'K_means_valid.csv'))
    df_test = pd.read_csv(os.path.join(dirPath, 'K_means_test.csv'))

    df_train = np.array(df_train)[:, 1:-1]
    df_valid = np.array(df_valid)[:, 1:]
    df_test = np.array(df_test)[:, 1:-1]

    return df_train, df_valid, df_test


"""
    Defining the Model
"""


# return the distance of two sample 'x' and 'y'
def Dist(x, y):
    return np.linalg.norm(x - y)


# return the centroid of a list of samples
def Centroid(list):
    return np.mean(list, axis=0)


# return the sum of variance of a cluster
def Variance(centroid, cluster):
    sum = 0
    for cl in cluster:
        sum += Dist(cl, centroid)
    return sum


# classify the test sample to a nearest center
def Classify(k, test, centroid):
    minIndex = -1
    minDist = float('inf')
    for i in range(k):
        if (Dist(test, centroid[i]) < minDist):
            minDist = Dist(test, centroid[i])
            minIndex = i
    return minIndex + 1


"""
    Defining the Loss Function/Optimization Algorithm
"""
# k-means doesn't use gradient descent, so I think this part isn't necessary(it is included in the training process)

"""
    Training
"""


def train(k, maxIter, df_train,visualize=False):
    rows = df_train.shape[0]
    var_best = [[float('inf')] for i in range(k)]
    center_best = clusterSet_best = [[] for i in range(k)]
    for iter in range(maxIter):
        cluster = np.zeros(rows) - 1  # cluster[i]=k : sample i belongs to the k-th cluster
        clusterSet = [[] for i in range(k)]
        center = np.array((random.sample(list(df_train), k)))

        change = True
        while (change == True):
            change = False
            # get the cluster of each sample, stored in cluster[]
            for i in range(rows):
                minDist = float('inf')
                minIndex = -1
                # get the nearest center for each i-th sample
                for j in range(k):
                    distance = Dist(df_train[i], center[j])
                    if (distance < minDist):
                        minDist, minIndex = distance, j
                # update the cluster of i-th sample
                if (cluster[i] != minIndex):
                    cluster[i] = minIndex
                    change = True
            # update the center
            for i in range(k):
                clusterSet[i] = [j for j in range(rows) if (cluster[j] == i)]
                center[i] = Centroid(df_train[clusterSet[i]])

            # draw the picture
            if visualize==True:
                draw(k, df_train, clusterSet)
                plt.pause(1)
                plt.clf()

        var_iter = [Variance(center[i], df_train[clusterSet[i]]) for i in range(k)]

        if (np.sum(var_iter) < np.sum(var_best)):
            var_best = var_iter
            center_best = center
            clusterSet_best = clusterSet
    if visualize == True:
        plt.ioff()
        draw(k, df_train, clusterSet_best)

    return center_best, var_best, clusterSet_best


"""
    Testing
"""


def test(k, df_test, center, mode, dirPath, inputFile, outputFile=None):
    n = df_test.shape[0]
    df = pd.read_csv(os.path.join(dirPath, inputFile))
    if (mode == 'write'):
        for i in range(n):
            df['labels'].at[i] = 'cluster' + str(Classify(k, df_test[i], center))
        df.to_csv(os.path.join(dirPath, outputFile))
    elif (mode == 'print'):
        for i in range(n):
            print('cluster', Classify(k, df_test[i], center))


"""
    Bonus
"""


def draw(k, df_train, clusterSet):
    tsne = TSNE(n_components=2, init='pca', random_state=1000)
    X_tsne = tsne.fit_transform(df_train)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    for i in range(k):
        for j in range(len(clusterSet[i])):
            plt.scatter(X_norm[clusterSet[i]][j][0], X_norm[clusterSet[i]][j][1], color=plt.cm.Set1(i))

    plt.show()


def relation(n_k, maxIters, df_train):
    varSet = []
    for i in range(1, n_k + 1):
        _, var_best, _ = train(i, maxIters, df_train,visualize=False)
        varSet.append(np.sum(var_best))
    plt.ioff()
    plt.plot([i for i in range(1, n_k + 1)], varSet)
    plt.xlabel('k')
    plt.ylabel('variance')
    plt.show()


if __name__ == '__main__':
    k = 3
    maxIters = 1
    path = r'K_means/'
    # Read date
    df_train, df_valid, df_test = Read(path)
    df_valid_test = df_valid[:, :-1]
    # Train
    # center_best, var_best, clusterSet_best = train(k, maxIters, df_train,visualize=False)
    # Test
    #test(k, df_test, center_best, 'write',path,'K_means_test.csv','test_result.csv')

    # Bonus1

    # draw(k, df_train, clusterSet_best)

    # Bonus2
    n_k = 10
    relation(n_k, maxIters, df_train)

# 下标形式和多加一点一列存储下标
# for循环用下标还是元素
# 代码用jupyter notebook写吗
# 迭代器
# 类
# 怎么分出类别
