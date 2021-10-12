#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:38:32 2021

@author: chanh
"""

xyscatter = [25,
24,
25,
25,
25,
25,
21,
24,
24,
24,
30,
29,
25,
25,
25,
25,
27,
29,
25,
29,
24,
29,
24,
25,
27,
25,
24,
27,
24,
25,
25,
24,
24,
25,
25,
30,
29,
29,
27,
30,
27,
24,
29,
25,
27,
25,
25,
25,
23,
29,
22,
30]

print(xyscatter)

xscatter=[]
yscatter=[]

x=0
while x < len(xyscatter):
    xscatter.append(xyscatter[x])
    x+=2
    
x=1
while x < len(xyscatter):
    yscatter.append(xyscatter[x])
    x+=2    
    
print(xscatter)
print(yscatter)

# Plot
# import matplotlib.pyplot as plt
# # plt.scatter(X_train[0], X_train[1], alpha=0.8) 
# # print(X_train[0], X_train[1])
# plt.scatter(xscatter, yscatter, alpha=0.8) 
# plt.title('Scatter plot')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()

contamination = 0.1  # percentage of outliers
n_train = 500  # number of training points
n_test = 500  # number of testing points
n_features = 25 # Number of features

import pandas as pd
from pyod.utils.data import generate_data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=n_features,
                  contamination=contamination,
                  random_state=42)

print(X_train)

X_train = pd.DataFrame(np.reshape[yscatter, xscatter],2x3)
X_test = pd.DataFrame(X_test)

print(X_train)