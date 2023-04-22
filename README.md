# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Keerthika N
RegisterNumber: 212221230049
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

#Visualizing the data

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

#Sigmoid Function

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train=np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
## Output:
* Array Value of x

![1](https://user-images.githubusercontent.com/93427089/233768988-79e4a0c2-ebce-45fa-9858-02d3f1e9d5f6.png)

* Array Value of y

![2](https://user-images.githubusercontent.com/93427089/233769144-5a1bf2e3-c7e6-4b59-8a20-e34066ba76da.png)

* Exam 1 - score graph

![3](https://user-images.githubusercontent.com/93427089/233769147-5aca62fc-ef5e-423c-8f71-a6cde08f435a.png)

* Sigmoid function graph

![4](https://user-images.githubusercontent.com/93427089/233769152-f5dd418a-16f9-4aef-a35b-576610c773ff.png)

* X_train_grad value

![5](https://user-images.githubusercontent.com/93427089/233769161-1c49d581-1a11-410e-bda3-892b7c09584f.png)

* Y_train_grad value

![6](https://user-images.githubusercontent.com/93427089/233769172-3f5a7ffe-1e0e-4e0b-93d8-1efd8f1af0fb.png)

* Print res.x

![7](https://user-images.githubusercontent.com/93427089/233769181-11cbf883-f7f5-4bc5-91a7-c4322553f757.png)

* Decision boundary - graph for exam score

![8](https://user-images.githubusercontent.com/93427089/233769187-679382c9-63e6-45ba-8585-a9ca43a03a0d.png)

* Proability value

![9](https://user-images.githubusercontent.com/93427089/233769204-25f71956-1c98-4fdd-986b-7176aa04806c.png)

* Prediction value of mean

![10](https://user-images.githubusercontent.com/93427089/233769214-ccebaa44-b705-426f-98e6-e233ba5b44b9.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

