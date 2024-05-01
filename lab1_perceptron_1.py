# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import collections

# загружаем и подготавляваем данные
df = pd.read_csv('C:\\Users\student\Downloads\data_err.csv')

df = df.iloc[np.random.permutation(len(df))] # случайная перестановка элементов массива по индексу
y = df.iloc[0:100, 4].values # в y находится последний столбец с видами ирисов (правильные ответы)
y = np.where(y == "Iris-setosa", 1, -1) # если y == ирис сетоза, y = 1, иначе y = -1
X = df.iloc[0:100, [0, 2]].values # в массиве Х выбрали только первые 2 столбца (т.е. будем обучать только по 2-м признакам)

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи (shape возвращает кол-во столбцов X) 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


# создаем матрицу весов скрытого слоя
Win = np.zeros((1+inputSize,hiddenSizes)) 
# пороги w0 задаем случайными числами
Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes))) 
# остальные веса  задаем случайно -1, 0 или 1 
Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes))) 

#Wout = np.zeros((1+hiddenSizes,outputSize))

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
 
# функция прямого прохода (предсказания) 
def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict
#f = predict([1,2])
# обучение
# у перцептрона Розенблатта обучаются только веса выходного слоя 
# как и раньше обучаем подавая по одному примеру и корректируем веса в случае ошибки

errors = []
eta = 0.01
count = 0

while (1): 
    #Проверка на зацикливание
    if (any([all(Wout == i) for i in errors])):
        print("Обучение завеpшено - зацикливание")
        break
        
    errors.append (Wout.copy())
    count = 1
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi) 
        
        #Проверка на отсутствие ошибок в обучении
        if (target - pr == 0):
            count += 1
            if (count == X.shape[0]):
                #print("1 Обучение завеpшено - нет ошибок")
                break
                      
        #если таргет и предикт (pr) совпали(получается 0: веса не меняются)
        #тогда меняем веса на выходном слое   
        
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
        Wout[0] += eta * (target - pr)
            
    if (count == X.shape[0]):
        print("Обучение завеpшено - нет ошибок")
        break
        
# посчитаем сколько ошибок делаем на всей выборке
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
pr, hidden = predict(X)
#поделили на 2, потому что все ошибки считались(как, например, 1-(-1) = 2. А ошибка то одна! надо делить на 2 поэтому)
#добавили abs, чтобы при счете ошибок(например,-2-(-2) не получился 0 и мы не потеряли ошибки, но конкретно в этой программе все признаки положительные, поэтому пофиг)
sum (abs(pr-y.reshape(-1, 1)))/2

# далее оформляем все это в виде отдельного класса neural.py
