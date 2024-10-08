---
title: "Feed Forward Neural Net"
date: 2023-08-27T14:45:09-05:00
draft: false
author: "Steven Lasch"
tags:
  - machine learning
  - neural networks
  - applied math
  - data viz
  - plotly
image: /images/neural_net.gif
description: "In this article, we will walk through an example of a binary classification problem that is more complex. For this approach, we will be implementing a simple feed-forward neural network to determine the classes."
toc: true
mathjax: true
plotly: true
---

In this article, we will walk through an example of a classification problem that is more complex. We will attempt to classify points within concentric circles. Linear or logistic regression algorithms are not suited for this kind of classification, so we need something more powerful.

For this approach, we will be implementing a simple feed-forward neural network to determine the classes. I would recommend using an environment such as Google Colab that can run Jupyter Notebooks since all of the code in this article is written for that format.

## Introduction

For clustering and classification purposes, we will be using the [`sklearn.datasets.make_circles()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) function. This function allows us to create a test dataset of points that we are aiming to classify. Blue points represent the outer circle, and red points represent the inner circle. 

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


X,Y = datasets.make_circles(n_samples=500,
                            random_state=1,
                            noise=.15,		# controls the spread of points
                            factor=.25)		# controls closeness of the circles

# plot x1 vs x2
plt.scatter(X[Y==0, 0], X[Y==0,1], color="blue")
plt.scatter(X[Y==1, 0], X[Y==1, 1], color="red")
plt.show()
```
<div style="text-align: center;">
  <img src="https://rawcdn.githack.com/s-lasch/portfolio/ca0a02197a7198f97c23dbf70348493e714aecb8/exampleSite/content/blogs/feed-forward-net/neural_network_example.png" />
</div>

> For this exercise, we will be predicting the probability that a given point is red.

## Neural Network Visually

Training a model to classify points based on this dataset must be done using a neural network. You have probably seen a graph representing a neural network before. The diagram below shows the flow of values from the input end to the output end. 

<div style="text-align: center;">
  <img src="https://rawcdn.githack.com/s-lasch/portfolio/ca0a02197a7198f97c23dbf70348493e714aecb8/exampleSite/content/blogs/feed-forward-net/neural_net_graph.png" />
</div>

We will have 2 inputs and a bias, a 4-node hidden layer using sigmoid activation function, and a single output: the probability that a given point is red. 

## Finding \\( \hat{y} \\)

But how can we standardize the outputs of our model to resemble a probability? To do this, we require an activation function to standardize our model output between a range of numbers. For this exercise, [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function#roperties:~:text=Thus%20the%20cumulative%20distribution%20functions%20for%20many%20common%20probability%20distributions%20are%20sigmoidal.) is a great option.

The reason this is a good choice is due to the output of the sigmoid function. The output lies between \\( 0 \\) and \\( 1 \\). Since our model will classify points based on a probability of being red, (which we will call \\( \hat{y} \\), pronounced "y hat") this funciton is ideal because probabilities can only exist between \\( 0 \\) and \\( 1 \\).

The reason we are using 4 sigmoid functions is quite simple. Think of it this way: what we need to do is create a sort of perimeter around the data points. That will determine a threshold of sorts, where once a value has passed in either the \\( x_1 \\) or \\( x_2 \\) direction, a value becomes red.

<div style="text-align: center;">
  <img src="https://rawcdn.githack.com/s-lasch/portfolio/91f0e7c9dfb29c84eed4f7fb7f32fc3a9b6fb698/exampleSite/content/blogs/feed-forward-net/neural_network_example_sigmoids.svg" width="60%"/>
</div>

## The heart of our algorithm
The core of our model is the perceptron. At its most basic level, a [perceptron](https://www.wikiwand.com/en/Perceptron#:~:text=It%20is%20a%20type%20of%20linear%20classifier%2C%20i.e.%20a%20classification%20algorithm%20that%20makes%20its%20predictions%20based%20on%20a%20linear%20predictor%20function%20combining%20a%20set%20of%20weights%20with%20the%20feature%20vector.) is a type of artificial neuron that involves the classification of two or more variables. Its specific use case is **linear classification**, where the artificial neuron aims to find a function that best describes the decision boundary between two variables. If we are then given a data value the model has never seen before, it can therefore determine which class the data point belongs to. 

## Neural network programmatically

Below is the generic template of a neural network with two hidden layers. 

```python
import torch
import torch.nn as nn
from torch.nn import Linear


class Model(nn.Module):

    def __init__(self, input_size, H, output_size):
        super().__init__()

        # input layer to hidden layer
        self.linear = torch.nn.Linear(input_size, H)
        
        # hidden layer to output layer
        self.linear2 = torch.nn.Linear(H, output_size)

    # forward pass
    def forward(self, x):
        # uses sigmoid to determine probabilities as it goes through both models
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x

    # make a prediction about a value x
    def predict(self, x): 
    	# pred is the probability of a positive result
        pred = self.forward(x) 
        
        return [1, pred] if pred >= .5 else [0, pred]
```

## Training the network

To train our model, first we must convert our datasets to column matrices. This is because ML largely relies on core features of linear algebra such as matrix multiplication, which requires that the dimensions of input matrices align. If you treat a dataset as a matrix where each column is a feature and each row is an observation, it becomes convenient to represent weights, gradients, or individual data points as column vectors. This ensures consistency in operations like dot products, matrix multiplication, and matrix transformations.

```python
X_tensor = torch.tensor(X).float()                  # already a column matrix
Y_tensor = torch.tensor(Y).reshape(-1,1).float()    # ML needs column matrices
```

Now for the main part, which is training a model on our dataset. The process we must follow is:

1. Set a random seed to ensure reproducibility.
2. Call on our `Model()` class to mimic the network diagram.
3. We use BCE as our loss algorithm, as it handles logarithms which are used in the sigmoid activation function.
4. We use the [`torch.optim.Adam()`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer because it changes learning rate dynamically rather than wasting resources to find a suitable value.
5. Train model for \\( x \\) number of epochs so long as the model doesn’t overfit or underfit data.

```python
# set seed for consistency of randoms
torch.manual_seed(1)

# recreate network schema
model = Model(input_size=2, H=4, output_size=1)

# training setup
epochs = 130
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.1)

# training model
for i in range(epochs):
    
    # training process
    optimizer.zero_grad()
    Yhat = model.forward(X_tensor)		# pass X data through the neural network
    loss = criterion(Yhat, Y_tensor)	# compare the predicted values with actual Y values
    loss.backward()						# find derivative
    optimizer.step()					# take the step

    # print model parameters each iteration
    print(i+1)
    print(loss)
    print(list(model.parameters()))
    print()
```

## Visualize the training process

Watching a command line feed is cool and all to see that our model is training, but wouldn't it be cooler if we could see how our model makes decisions in real time?

Here is a GIF that shows the visualization of the training process. It uses a [`contourf()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html) plot to show the decision boundaries between the red and blue classes. Areas with lighter color represent a reduced probability that a given data point is either red or blue, though any value with probability \\( P \ge 0.5 \\) is considered red.

<div style="text-align: center;">
  <img src="https://rawcdn.githack.com/s-lasch/portfolio/721b15eab2e28813d08be5ba7f167554ff348163/exampleSite/content/blogs/feed-forward-net/neural_network_example.gif" />
</div>

Pretty cool, right?

The last step is to ask the model to classify a new point that it has never seen before. We’ll use the black point at \\( (-0.5, -0.4) \\) for this. 

```python
# make a prediction
print(model.predict(torch.tensor([-0.5, -0.4])))
```

```text
[1, tensor([0.6090], grad_fn=<SigmoidBackward0>)]
```

This means that our model predicted `True` for the black point. Recall, that our prediction, \\( \hat{y} \\), represents the probability that a given point is of the red class. With this in mind, the black point is to be put in the red class, with a 60.9% probability. This value is only moderately greater than 50%, representing a moderately confident prediction. This is also apparent visually, with only a slight red coloring on the graph around the black point.

## Completed Code

Here is the completed code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch
import torch.nn as nn
from torch.nn import Linear

#############################
### DEFINE NEURAL NETWORK ###
#############################
class Model(nn.Module):

    def __init__(self, input_size, H, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, H)
        self.linear2 = torch.nn.Linear(H, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    
    def predict(self, x): 
        pred = self.forward(x)
        return [1, pred] if pred >= .5 else [0, pred]
    
######################
### CREATE DATASET ###
######################
X,Y = datasets.make_circles(n_samples=500,
                            random_state=1,
                            noise=.15,		# controls the spread of points
                            factor=.25)		# controls closeness of the circles

X_tensor = torch.tensor(X).float()                  # already a column matrix
Y_tensor = torch.tensor(Y).reshape(-1,1).float()    # ML needs column matrices

######################
### MODEL TRAINING ###
######################
torch.manual_seed(1)

model = Model(input_size=2, H=4, output_size=1)

# for plotting purposes
x1 = np.arange(-1.5, 1.5, .1)
x2 = np.arange(-1.5, 1.5, .1)

rows, columns = np.meshgrid(x1, x2)

x = np.hstack([rows.ravel().reshape(-1,1), columns.ravel().reshape(-1,1)])
x_tensor = torch.tensor(x).float()

# training setup
epochs = 130
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.1)

# model training
for i in range(epochs):
    yhat = model.forward(x_tensor).float()
    yhat = yhat.reshape(rows.shape)

    plt.scatter(X[Y==0, 0], X[Y==0,1], color="blue")
    plt.scatter(X[Y==1, 0], X[Y==1, 1], color="red")
    plt.contourf(rows, columns, yhat.detach(), alpha=.4, cmap="RdBu_r")
    plt.scatter(-0.5, -0.4, color="black")
    plt.show()

    optimizer.zero_grad()
    Yhat = model.forward(X_tensor)
    loss = criterion(Yhat, Y_tensor)
    loss.backward()
    optimizer.step()
```
