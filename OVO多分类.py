# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:28:20 2020

@author: Micalry Xavier
"""

import  numpy as np
import matplotlib.pyplot as plt
def plot_decision_boundary(model,axis):  #两个数据特征基础下输出决策边界函数
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1,1)
        )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    cus=ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
    plt.contourf(x0,x1,zz,cmap=cus)

#采用iris数据集的两个数据特征进行模型训练与验证
from sklearn import datasets
d=datasets.load_iris()
x=d.data[:,:2]  #选取特征数据集的前两个数据特征，方便输出决策出边界进行训练结果的对比
y=d.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)
from sklearn.linear_model import LogisticRegression

#OVR方式的调用-默认方式
log_reg=LogisticRegression()  #不输入参数时，默认情况下是OVR方式
log_reg.fit(x_train,y_train)
print(log_reg.score(x_test,y_test))
plot_decision_boundary(log_reg,axis=[4,9,1,5])
plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.scatter(x[y==2,0],x[y==2,1],color="b")
plt.show()

#OVO的方式进行逻辑回归函数参数的定义，结果明显好于OVR方式
log_reg1=LogisticRegression(multi_class="multinomial",solver="newton-cg")
log_reg1.fit(x_train,y_train)
print(log_reg1.score(x_test,y_test))
plot_decision_boundary(log_reg1,axis=[4,9,1,5])
plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.scatter(x[y==2,0],x[y==2,1],color="b")
plt.show()

#采用iris数据的所有特征数据
x=d.data
y=d.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)
from sklearn.linear_model import LogisticRegression

#OVR方式的调用-默认胡方式
log_reg=LogisticRegression()  #不输入参数时，默认情况下是OVR方式
log_reg.fit(x_train,y_train)
print(log_reg.score(x_test,y_test))

#采用OVO的方式进行逻辑回归函数参数的定义，结果明显好于OVR方式
log_reg1=LogisticRegression(multi_class="multinomial",solver="newton-cg")
log_reg1.fit(x_train,y_train)
print(log_reg1.score(x_test,y_test))