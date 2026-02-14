# DecisionTreeClassifier

## Overview:
This repository contains purely python implementation of a Decision Tree Algorithm.
Decision Trees(DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
This repository contrains the logic of building this algorithm including how the decision is made for the new data points.


## Core characteristics:
### It can handle-
Binary classification
Multi-class classification
Numerical and categorical features

### No need for One-hot-encoding
This code builds the algorithm with numerical as well as categorical data. This way we are not loosing any context of the categorical features
It builds the tree looking at the strings and numbers seperately, like human make decisions

### Selects the best feature to split the data using splitting criterion such as:
Entropy (Information Gain)

### It Recursively splits the dataset until:
Maximum depth is reached
Minimum samples per leaf is reached
Data becomes pure

### Evaluation Metrics coded from scratch:
Confusion Matrix
Accuracy
Precision
Recall
F1 score
FPR
Specificity


## Mathematics it uses:

### 📐 Entropy

Entropy represents the impurity of a node.

$$
Entropy = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

Where:
- \( C \) = number of classes
- \( p_i \) = probability of class i


### 📐 Information Gain

Information Gain measures how much uncertainty (entropy) is reduced after a split.

$$
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
$$

Where:

- \( S \) = original dataset  
- \( A \) = feature used for splitting  
- \( S_v \) = subset of S where feature A takes value v  
- \( |S| \) = total number of samples  
- \( |S_v| \) = number of samples in subset  
- \( H(S) \) = entropy of dataset S


## How to Run:
### Clone the Repository

```bash
git clone https://github.com/towardsyourgoal/DecisionTreeClassifier_from_Scratch.git
cd DecisionTreeClassifier_from_Scratch
pip install -r requirements.txt
```

## Tech-stack:
Python
Numpy
