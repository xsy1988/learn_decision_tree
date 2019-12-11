# Create by MrZhang on 2019-11-19

import numpy as np
from math import log
import operator
import pickle

# 计算当前数据集的熵
def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        entropy -= prob * log(prob, 2)

    return entropy

# 对当前数据集进行一次划分
def splitDataSet(dataSet, index, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 寻找对当前数据集进行划分对最佳方式：按照某个特征划分数据集后信息增益最大
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    bestEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcEntropy(subDataSet)
        infoGain = bestEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    # 设置判断语句，在满足判断条件时，跳出底下对递归循环
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 当前数据集，找出最佳划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeat]

    myTree = {bestFeatureLabel: {}}
    # 已经使用过的特征，从labels列表中删除，不再重复出现或使用
    del(labels[bestFeat])

    # 按照当前最佳特征，对数据集进行划分后，使用递归函数，对划分后对数据子集，分别再进行划分
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDic = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDic.keys():
        if testVec[featIndex] == key:
            if type(secondDic[key]).__name__ == 'dict':
                classLabel = classify(secondDic[key], featLabels, testVec)
            else:
                classLabel = secondDic[key]
    return classLabel

#  存储Tree
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

# 读取Tree
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

def createDataSet():
    dataSet1 = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [0, 0, 0, 0, 'no'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels1 = ['年龄', '有工作', '有房子', '信用']
    return dataSet1, labels1

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    _, featureLabels = createDataSet()
    myTree = createTree(dataSet, labels)
    print(myTree)
    testVce = [2, 0, 0, 1]
    result = classify(myTree, featureLabels, testVce)
    print("是否放贷：", result)
