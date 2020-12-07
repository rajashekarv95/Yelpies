import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import 
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
from xgboost import XGBClassifier

# Helper functions for encoding 
# Lebel encoding

le = preprocessing.LabelEncoder()

def encoded_cols(df):
  cols_to_encode = []
  for col in df.columns:
    if df[col].dtype == object:
      cols_to_encode.append(col)

  return cols_to_encode

def label_encoding(df, cols_to_encode):
   le.fit(df[cols_to_encode].values.flatten())
   df[cols_to_encode] = df[cols_to_encode].apply(le.fit_transform)
   return df
  
def mapping(df, cols_to_encode):
   mapping = {}
   for i in cols_to_encode:
    mapping.update(dict(zip(le.classes_, range(len(le.classes_)))))
    return mapping

def apply_map_to_test(df_test, cols_to_encode):
  for col in cols_to_encode:
    df_test[col] = df_test[col].apply(lambda x: mapping.get(x))
    return df_test


## One hot encoding

def one_hot(df, cols_to_encode):
  df_subset_cat = df[cols_to_encode]
  df_subset_num = df[df.columns.difference(cols_to_encode)]

  df_subset_cat_ohe = pd.get_dummies(df_subset_cat)
  df_col_merged = pd.concat([df_subset_cat_ohe, df_subset_num], axis=1)

  return df_col_merged

## Read processed data 

df = pd.read_csv(cleaned_data)
reviews = pd.read_csv(reviews_file)

## map restaurants with reviews  and get relevant columns 
df_reviews_sub = df_reviews[['camis','boro','zipcode','cuisine_description','inspection_date_date','action','review_wo_stop']]
df_reviews = df.merge(reviews, on = ['camis','inspection_date_date'], how = 'inner')

## Get average polarity scores and ratings per restaurant 
avg_polarity = pd.read_csv(polarity_file)

# Map polarity scores to each restaurant
avg_polarity_sub = avg_polarity[['camis','polarity_no_stopwords','rating','inspection_date_date']]
df_reviews_sub = df_reviews_sub.merge(avg_polarity_sub, on = ['camis','inspection_date_date'], how = 'left')
df_reviews_sub.fillna(0, inplace=True)

# convert target variable to binary 
df_reviews_sub = df_reviews_sub.replace({'No violations were recorded at the time of this inspection.':0, 'Establishment Closed by DOHMH.  Violations were cited in the following area(s) and those requiring immediate action were addressed.':1})

## Separate out standard features from text reviews 
df_reviews_ohe = df_reviews_sub[['boro','zipcode','cuisine_description','polarity_no_stopwords','rating','action']]
df_reviews_token = df_reviews_sub['review_wo_stop']

## One hot encode standard categorical features
cols_to_encode = encoded_cols(df_reviews_ohe)
encoded_data = one_hot(df_reviews_ohe, cols_to_encode)



