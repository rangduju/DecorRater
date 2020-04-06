# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:08:34 2020

@author: Micalry Xavier
"""

from sklearn import datasets



import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

class LogisticRegression:

   def __init__(self, X, Y, alpha, theta, num_iters):

       self.X = X #训练集样本
       self.Y = Y #训练集标签
       self.alpha = alpha #学习率
       self.num_iters = num_iters #迭代次数
       self.theta = theta #权重w
       self.N = len(self.Y) #训练集样本数

def train(self):
       for index in range(self.num_iters):
           new_theta = self.__Gradient_Descent() 
           self.theta = new_theta #更新权重
           if index % 100 == 0:
               print ('cost is：', self.__Cost_Function())
               
               
def __Cost_Function(self):
       sumOfErrors = 0 #记录所有样本的损失
       for i in range(self.N): #遍历每个样本
           x_i = self.X[i]
           y_i = self.Y[i]
           h_i = self.__Hypothesis(x_i) #即公式4中的h(x)函数
           error = Y[i] * math.log(h_i) + (1-Y[i]) * math.log(1-h_i)
           sumOfErrors += error
       J = -1/float(self.N) * sumOfErrors #公式5
       return J               
               
def __Hypothesis(self, x):
       """对一个样本x，计算h(x)的值"""
       z = 0
       for i in range(len(self.theta)):
           z += x[i] * self.theta[i]
       return self.__Sigmoid(z)

def __Sigmoid(self, z):
       Sig = float(1.0 / float((1.0 + math.exp(-1.0*z))))
       return Sig               


def __Gradient_Descent(self):
       new_theta = []
       for j in range(len(self.theta)):
           CFDerivative = self.__Cost_Function_Derivative(j)
           new_theta_value = self.theta[j] - CFDerivative #即公式6
           new_theta.append(new_theta_value)
       return new_theta
   
    
def __Cost_Function_Derivative(self,j):
       sumErrors = 0
       for i in range(self.N):
           x_i = self.X[i]
           x_ij = x_i[j]
           h_i = self.__Hypothesis(x_i)
           error = (h_i - self.Y[i]) * x_ij
           sumErrors += error
       J = float(self.alpha)/float(self.N) * sumErrors
       return J




def test(self, X_test, Y_test):
       score = 0
       length = len(X_test)
       for i in range(length):
           prediction = round(self.__Hypothesis(X_test[i]))
           if prediction == Y_test[i]:
               score += 1
       acc = float(score) / float(length)
       print ("acc is: ", acc)    
       
       
       
       
       
#读入数据       
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv("data.csv", header=0)

df.columns = ["grade1","grade2","label"]

x = df["label"].map(lambda x: float(x.rstrip(';')))

X = df[["grade1","grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)


initial_theta = [0,0]
alpha = 0.1
iterations = 1000
clf = LogisticRegression(X_train, Y_train, alpha, initial_theta, iterations)
clf.train()
clf.test(X_test, Y_test)
       
       