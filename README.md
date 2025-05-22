<H3>Name: Vishal S</H3>
<H3>Register no: 212223110063</H3>
<H3>Date: </H3>
<H3>Experiment No. 2 </H3>

# Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
### Importing packages:
```
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
```
###  Data loading and plotting:
```
iris=load_iris()
X=iris.data
Y=iris.target

X=X[Y!=2]
Y=Y[Y!=2]

plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',edgecolors='k')
plt.title("linearly seperable data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
### Data scaling and splitting:
```
scaler=StandardScaler()
x_scale=scaler.fit_transform(X)
xtrain,xtest,ytrain,ytest=train_test_split(x_scale,Y,test_size=0.2)

xtrain=np.c_[np.ones(xtrain.shape[0]),xtrain]
xtest=np.c_[np.ones(xtest.shape[0]),xtest]

ytrain=np.where(ytrain==0,-1,1)
ytest=np.where(ytest==0,-1,1)
```
### Perceptron training and plotting:
```
w=np.zeros(xtrain.shape[1])
learning_rate=0.01
errors=[]
iter=100

for epochs in range(iter):
  total_error=0
  for i in range(len(xtrain)):
    xi=xtrain[i]
    yi=ytrain[i]
    prediction=np.sign(np.dot(w,xi))
    prediction=prediction if prediction !=0 else -1
    error=yi-prediction
    w+=learning_rate*error*xi
    total_error+=int(error!=0)
  errors.append(total_error)

plt.plot(range(iter), errors, marker='o')
plt.title("Errors per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of misclassifications")
plt.grid(True)
plt.show()
```
### Testing:
```
def predict(X, W):
    predictions = np.sign(np.dot(X, W))
    predictions[predictions == 0] = -1
    return predictions

Ypred = predict(xtest, w)
accuracy = np.mean(Ypred == ytest)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
```
# OUTPUT:
### linearly seperable data plot:
![image](https://github.com/user-attachments/assets/62f933b6-de43-4fdd-b615-a5676e97d48d)
### error plot:
![image](https://github.com/user-attachments/assets/3b281450-b52f-415a-839e-c1c8ae3be368)

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
