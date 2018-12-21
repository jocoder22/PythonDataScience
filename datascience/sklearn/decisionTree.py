#!/usr/bin/env python

import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as splitit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression