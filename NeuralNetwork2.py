import sys
import csv
import math
import copy
import numbers
import numpy as np
from numpy.core.multiarray import dtype
from sklearn import preprocessing

def parse_train_labels(labels):
    lines = labels.readlines()
    output = []
    for line in lines:
        output.append(line.replace("\r", "").replace("\n", "").replace("yes", "1").replace("no", "0"))

    return output



def process_attributes(attributes):
    output = []
    for line in attributes:
        updatedLine = []
        for element in line:
            updatedLine.append(element.replace("yes", "1").replace("no", "0"))
        output.append(updatedLine)
    return output


def adjust_full_weight(attributes, label, inputHiddenWeights, hiddenFinalWeights):
    learn = 0.01
    outH = []
    netY = 0

    input = []
    for i in range(1, len(attributes)):
        input.append([])
        input[i - 1] = copy.deepcopy(attributes[i])
        input[i - 1].insert(0, 1)
    InputArray=np.array(input,dtype=float)
    HiddenWeights=np.array(inputHiddenWeights,dtype=float)

    netH = np.dot(InputArray,HiddenWeights)

    outH = []


    for i in range(len(netH)):
        outH.append([1.0])
        for j in range(len(netH[i])):
            e = 1
            try:
                e = math.exp(-netH[i][j])
            except OverflowError:
                e = float('inf')
            outH[i].append(1 / (1 + e))
    outH=np.array(outH,dtype=float)
    hiddenFinalWeights=np.array(hiddenFinalWeights,dtype=float)

    netY = np.dot(outH, hiddenFinalWeights)

    outY = []
    sum = 0

    for i in range(len(netY)):
        outY.append([])
        outY[i].append(1 / (1 + math.exp(-netY[i][0])))
        sum = sum + (outY[i][0] - float(label[i])) ** 2

    dk = []
    for i in range(len(outY)):
        dk.append([])
        dk[i].append(outY[i][0] * (1 - outY[i][0]) * (float(label[i]) - outY[i][0]))

    dh = []
    #print outH
    for i in range(len(outH)):
        dh.append([])
        for j in range(1, len(outH[i])):
            dh[i].append(outH[i][j] * (1 - outH[i][j]) * float(hiddenFinalWeights[j][0]) * dk[i][0])

    #print outH
    outH = np.transpose(outH)

    #print dk
    #print outH
    outH=np.array(outH,dtype=float)
    dk=np.array(dk,dtype=float)

    mul = np.dot(outH, dk)

    #print mul
    for i in range(len(mul)):
        hiddenFinalWeights[i][0] = float(hiddenFinalWeights[i][0]) + learn * mul[i][0]

    #print hiddenFinalWeights

    #print input
    #print dh
    input2 = np.transpose(input)
    input2=np.array(input2,dtype=float)
    dh=np.array(dh,dtype=float)
    mul = np.dot(input2, dh)
    #print mul
    for i in range(len(inputHiddenWeights)):
        for j in range(len(inputHiddenWeights[0])):
            inputHiddenWeights[i][j] = float(inputHiddenWeights[i][j]) + learn * mul[i][j]

    return sum

def predict(attributes, inputHiddenWeights, hiddenFinalWeights):
    outH = []
    netY = 0

    input = []
    for i in range(1, len(attributes)):
        input.append([])
        input[i - 1] = copy.deepcopy(attributes[i])
        input[i - 1].insert(0, 1)
    input=np.array(input,dtype=float)
    inputHiddenWeights=np.array(inputHiddenWeights,dtype=float)

    netH = np.dot(input, inputHiddenWeights)
    outH = []

    #print netH
    for i in range(len(netH)):
        outH.append([1.0])
        for j in range(len(netH[i])):
            e = 1
            try:
                e = math.exp(-netH[i][j])
            except OverflowError:
                e = float('inf')
            outH[i].append(1 / (1 + e))
    outH=np.array(outH,dtype=float)
    hiddenFinalWeights=np.array(hiddenFinalWeights,dtype=float)

    netY = np.dot(outH, hiddenFinalWeights)

    for i in range(len(netY)):
        outY = 1 / (1 + math.exp(-netY[i][0]))
        # if outY > 0.5:
        #     print('yes')
        # else:
        #     print('no')
        print(outY)

def adjust_weight(attributes, label, inputHiddenWeights, hiddenFinalWeights):
    learn = 0.4
    netH = []
    outH = []
    netY = 0

    input = copy.deepcopy(attributes)
    input.insert(0, 1)

    #print input

    for i in range(len(inputHiddenWeights[0])):
        net = 0
        for j in range(len(inputHiddenWeights)):
            net = net + float(inputHiddenWeights[j][i]) * float(input[j])
        netH.append(net)

    #print netH

    outH.append(1)
    for i in range(len(netH)):
        e = 1
        try:
            e = math.exp(-netH[i])
        except OverflowError:
            e = float('inf')
        outH.append(1 / (1 + e))

    #print outH
    for i in range(len(outH)):
        netY = netY + outH[i] * float(hiddenFinalWeights[i][0])

    outY = 1 / (1 + math.exp(-netY))
    loss = (outY - float(label)) ** 2

    dh = []
    dk = outY * (1 - outY) * (float(label) - outY)

    for i in range(1, len(outH)):
        dh.append(outH[i] * (1 - outH[i]) * float(hiddenFinalWeights[i][0]) * dk)

    for i in range(len(hiddenFinalWeights)):
        hiddenFinalWeights[i][0] = float(hiddenFinalWeights[i][0]) + learn * dk * outH[i]

    for i in range(len(inputHiddenWeights)):
        for j in range(len(inputHiddenWeights[0])):
            inputHiddenWeights[i][j] = float(inputHiddenWeights[i][j]) + learn * float(input[i]) * dh[j]

    return loss

def normalize(attributes,j):
    #year = []
    x_array=[]
    tlist = list(zip(*attributes))
    year=tlist[j]

    for i in range(1, len(year)):
        x_array.append(float(year[i]))

    normalized_X = preprocessing.normalize([x_array])

    for i in range(1, len(year)):
        attributes[i][j]=normalized_X[0][i-1]
    return  attributes

def normalize_key(attributes,j):
    list = []
    OutputLabel=[]

    for i in range(0, len(attributes)):

        list.append(float(attributes[i]))

    minVal = min(list)
    maxVal = max(list)

    for i in range(0, len(attributes)):
        OutputLabel.append((float(attributes[i]) - minVal) / (maxVal - minVal))
    return OutputLabel

if len(sys.argv) == 5:
    exit(0)
else:
    try:
        trainLabels = []
        attributes = []
        testFile = []
        inputHiddenWeights = []
        hiddenFinalWeights = []
        labelOutput = []
        attributes_file = open('C:/Users/cool dude/PycharmProjects/hello/NeuralNetwork/Data/education_train.csv')
        attributes = csv.reader(attributes_file, delimiter=',')

        attributes = process_attributes(attributes)


        labelsFile = open('C:/Users/cool dude/PycharmProjects/hello/NeuralNetwork/Data/education_train_keys.txt', "r")
        labels = parse_train_labels(labelsFile)
        labelOutput=normalize_key(labels,0)

        test_file = open('C:/Users/cool dude/PycharmProjects/hello/NeuralNetwork/Data/education_dev.csv')
        testFile = csv.reader(test_file, delimiter=',')
        testFile = process_attributes(testFile)

        inputHiddenWFile = open('C:/Users/cool dude/PycharmProjects/hello/NeuralNetwork/Data/education_weights_1.csv')
        IHWeights = csv.reader(inputHiddenWFile, delimiter=',')

        for weight in IHWeights:
            inputHiddenWeights.append(weight)

        hiddenFinalWFile = open('C:/Users/cool dude/PycharmProjects/hello/NeuralNetwork/Data/education_weights_2.csv')
        HFWeights = csv.reader(hiddenFinalWFile, delimiter=',')

        for weight in HFWeights:
            hiddenFinalWeights.append(weight)

        attributes = normalize(attributes,0)
        attributes = normalize(attributes, 1)
        attributes = normalize(attributes, 2)
        attributes = normalize(attributes, 3)
        attributes = normalize(attributes, 4)

        inputHiddenWeights2 = copy.deepcopy(inputHiddenWeights)
        hiddenFinalWeights2 = copy.deepcopy(hiddenFinalWeights)

        iterations = 600
        loss = 0
        for i in range(iterations):
            loss = adjust_full_weight(attributes, labelOutput, inputHiddenWeights, hiddenFinalWeights)
            if i < 15:
                print (loss / 2)

        print("GRADIENT DESCENT TRAINING COMPLETED!")

        iterations = 15

        for i in range(iterations):
            loss2 = 0
            for iterator in range(1, len(attributes)):
                loss2 = loss2 + adjust_weight(attributes[iterator], labelOutput[iterator - 1], inputHiddenWeights2, hiddenFinalWeights2)
            print (loss2 / 2)

        print("STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.")

        normalize(testFile, 0)
        normalize(testFile, 1)
        normalize(testFile, 2)
        normalize(testFile, 3)
        normalize(testFile, 4)


        predict(testFile, inputHiddenWeights, hiddenFinalWeights)

    finally:
        labelsFile.close
