# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:29:37 2021

@author: AM4
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))] # случайная перестановка элементов массива по индексу

# возьмем первые 100 строк, 4-й столбец 
y = df.iloc[:, 4].values
# так как ответы у нас строки - нужно перейти к численным значениям

y_ = np.zeros((150,3))
for i in range (150):
    for j in range(0, 3):
        if (y[i] == "Iris-setosa"):
            y_[i][0] = 1
        if (y[i] == "Iris-versicolor"):
            y_[i][1] = 1
        if (y[i] == "Iris-virginica"):
            y_[i][2] = 1
            
#создание списка
# y_ = [[0]*3 for i in range (150)]

# возьмем два признака, чтобы было удобне визуализировать задачу
X = df.iloc[:, [0,1,2,3]].values

# добавим фиктивный признак для удобства матричных вычислений
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

# зададим функцию активации - сигмоида
def sigmoid(y):
    return 1 / (1 + np.exp(-y))
    #return 1 / (1 + np.exp(-y, dtype = ))

# нам понадобится производная от сигмоиды при вычислении градиента
def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))


# инициализируем нейронную сеть 
inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 5 # задаем число нейронов скрытого слоя 
outputSize = 3 if len(y_.shape) else y_.shape[1] # количество выходных сигналов равно количеству классов задачи

# веса инициализируем случайными числами, но теперь будем хранить их списком
weights = [
    np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),  # веса скрытого слоя
    np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))  # веса выходного слоя
]

# прямой проход 
def feed_forward(x):
    #чтобы в массиве было 2 размерности
    input_ = np.zeros((1, x.shape[0]))
    for u in range(x.shape[0]):
        input_[0][u] = x[u]
    
    #input_ = x # входные сигналы

    hidden_ = sigmoid(np.dot(input_, weights[0])) # выход скрытого слоя = сигмоида(входные сигналы*веса скрытого слоя)
    output_ = sigmoid(np.dot(hidden_, weights[1]))# выход сети (последнего слоя) = сигмоида(выход скрытого слоя*веса выходного слоя)

    # возвращаем все выходы, они нам понадобятся при обратном проходе
    return [input_, hidden_, output_]


# backprop собственной персоной
# на вход принимает скорость обучения, реальные ответы, предсказанные сетью ответы и выходы всех слоев после прямого прохода
def backward(learning_rate, target, net_output, layers):
    
    target1 = np.zeros((1, target.shape[0]))
    for u in range(target.shape[0]):
        target1[0][u] = target[u]
    # считаем производную ошибки сети
    err = (target1 - net_output)

    # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
    # для этого используем chain rule
    # цикл перебирает слои от последнего к первому
    for i in range(len(layers)-1, 0, -1):
        # градиент слоя = ошибка слоя * производную функции активации * на входные сигналы слоя
        
        # ошибка слоя * производную функции активации
        err_delta = err * derivative_sigmoid(layers[i])       
        
        # пробрасываем ошибку на предыдущий слой
        err = np.dot(err_delta, weights[i - 1].T)
        
        # ошибка слоя * производную функции активации * на входные сигналы слоя
        # b = np.zeros((1,layers[i - 1].shape[0]))
        # for j in range (layers[i - 1].shape[0]):
        #         b[0][j] = layers[i - 1][j]
                
        dw = np.dot(layers[i - 1].T, err_delta)
        #err_delta = np.zeros((1,layers[i - 1].shape[0]))
                
        # обновляем веса слоя
        weights[i - 1] += np.dot(learning_rate, dw)
        
        

# функция обучения чередует прямой и обратный проход
def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward(learning_rate, target, output[2], output)
    return None

# функция предсказания возвращает только выход последнего слоя
def predict(x_values):
    return feed_forward(x_values)[-1]

summ = []
# задаем параметры обучения
iterations = 150
learning_rate = 0.001

# обучаем сеть (фактически сеть это вектор весов weights)
for p in range(iterations):
    df = df.iloc[np.random.permutation(len(df))]
    y = df.iloc[:, 4].values
    
    y_ = np.zeros((150,3))
    for i in range (150):
        for j in range(0, 3):
            if (y[i] == "Iris-setosa"):
                y_[i][0] = 1
            if (y[i] == "Iris-versicolor"):
                
                y_[i][1] = 1
            if (y[i] == "Iris-virginica"):
                y_[i][2] = 1
                
    X = df.iloc[:, [0,1,2,3]].values

    # добавим фиктивный признак для удобства матричных вычислений
    X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

    #корректируем веса на каждой итерации по количеству примеров в выборке
    for t in range (0, X.shape[0]):
        train(X[t,:], y_[t,:], learning_rate)
    
    #train(X[0,:], y_[0,:], learning_rate)
    
    if p % 10 == 0:
        for h in range (0, X.shape[0]):
            summ.append(predict(X[h,:]))
        print("На итерации: " + str(p) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y_- summ))))
        summ = []
   
# считаем ошибку на обучающей выборке

# pr = []
# for t in range (0, X.shape[0]):
#     pr.append(predict(X[t,:]))
# print(sum(abs(y_-pr)))
    
# считаем ошибку на всей выборке
y = df.iloc[:, 4].values

y_ = np.zeros((150,3))
for i in range (150):
    for j in range(0, 3):
        if (y[i] == "Iris-setosa"):
            y_[i][0] = 1
        if (y[i] == "Iris-versicolor"):
            y_[i][1] = 1
        if (y[i] == "Iris-virginica"):
            y_[i][2] = 1

X = df.iloc[:, [0,1,2,3]].values
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

# for k in range (0, X.shape[0]):
#     pr = predict(X[k,:])
#     print(sum(abs(y_[k]-pr)))


