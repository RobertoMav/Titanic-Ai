#!/usr/bin/env python
# coding: utf-8

# # Code Modularity

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[3]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


# In[4]:


def NN_pred(yhat):
    if yhat >= 0.5:
        return 1
    else:
        return 0


# In[5]:


def eval_err(y, yhat):
    m = y.shape[0]
    incorrect = 0
    y = y.tolist()
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
            
    incorrect = incorrect / m
    
    return incorrect 


# In[6]:


def pred_output(prediction):
    ex = prediction.shape[0]
    output = []
    for i in range(ex):
        output.append(NN_pred(prediction[i]))
    
    return output


# In[7]:


def sigmoid(z):

    calc = math.e**-z
    g = 1 / (1 + calc)

    return g

