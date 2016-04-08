# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 14:37:17 2016

@author: YI
"""


#Linear regression practice 
import pandas as pd
import numpy as np

#load data
a=pd.read_csv('ex1data1.txt',header=None)    
X=a[0]
y=a[1]
m=len(y)
b=np.asarray(a[0])  #convert X to array
b=b.reshape(m,1)
y=np.asarray(a[1])  #convert y to array
y=y.reshape(m,1)
#Loss function
##initialize parameters
theta=np.zeros((2,1))
num_iters=1500
alpha=0.01          #learning rate
ones=np.ones((m,1))
X=np.concatenate((ones,b),axis=1)  #97*2
##create a function
def loss(X,y,theta):
    ##loss function
    L=np.sum((X.dot(theta)-y)**2)/(2*m)
    return L
       
#create a gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    
    L_history = np.zeros((num_iters, 1))

    for i in xrange(num_iters):
        
        dtheta0 = np.sum(X.dot(theta)-y)/m
        dtheta1 = np.sum((X.dot(theta)-y)*b)/m

        theta[0][0] = theta[0][0] - alpha * dtheta0
        theta[1][0] = theta[1][0] - alpha * dtheta1

        L_history[i, 0] = loss(X, y, theta)

    return theta, L_history
    
gradient_descent(X,y,theta,alpha,num_iters)

X1=np.array((1,3.5))
predict1 = X1.dot(theta)
print predict1

X2=np.array((1,7.0))
predict2 = X2.dot(theta)
print predict2