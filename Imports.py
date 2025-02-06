# Description: This file contains the imports of the libraries that will be used in throughout the notebooks created for the code part of the master thesis.
# Not all imports will be necessary for all notebooks.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import pickle
import re
import numbers
from openpyxl import load_workbook
import time

from typing import List, Union
from tqdm import tqdm
from rapidfuzz import process, fuzz

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SMOTENC, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import scipy.stats as stats

sns.set_style("white")
font_title = {'family': 'Calibri', 'color':  'black', 'weight': 'normal', 'size': 20}
font_label = {'family': 'Calibri', 'color':  'black', 'weight': 'normal', 'size': 13}

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)