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
from sklearn.decomposition import PCA


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

## Encode text data with n_gram = 2
vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),stop_words='english',binary=True)

X2_train = vectorizer2.fit_transform(df_reviews_token)
X2_train = X2_train.todense()
X2_train_pd = pd.DataFrame(X2_train)

### concat X2_train to encoded_data

encoded_data = pd.concat([encoded_data, X2_train_pd], axis = 1)

## Split this entire dataset into train-test


train, test = train_test_split(encoded_data, test_size=0.25)

x_train = train.drop(columns='action')
y_train = train['action']
x_test = test.drop(columns='action')
y_test = test['action']

### Run different models and get AUC graph 

def train_models(x_train, y_train, x_test, y_test, model_type):
  if model_type == 'lr':
    model = LogisticRegression(random_state=123).fit(x_train, y_train)
  elif model_type == 'svm':
    model = SVC(kernel="poly", probability=True, random_state=123).fit(x_train, y_train)
  elif model_type == 'dt':
    model = DecisionTreeClassifier(random_state=123, criterion='entropy').fit(x_train, y_train)
  elif model_type == 'rf':
    model = RandomForestClassifier(random_state=123).fit(x_train, y_train)
  elif model_type == 'gb':
    model = GradientBoostingClassifier(random_state=123).fit(x_train, y_train)
  elif model_type == 'xgb':
    model = XGBClassifier().fit(x_train, y_train)

  model_probs = model.predict_proba(x_test)[:, 1]
  model_pred = model.predict(x_test)
  fpr, tpr, _ = roc_curve(y_test, model_probs)
  auc = roc_auc_score(y_test, model_probs)
  f1 = f1_score(y_test, model_pred)
  precision = precision_score(y_test, model_pred)
  recall = recall_score(y_test, model_pred)

  return fpr, tpr, auc, f1, precision, recall

models = ['lr', 'svm', 'dt', 'rf', 'gb', 'xgb']
for model in models:
  fpr, tpr, auc, f1, precision, recall = train_models(x_train, y_train, x_test, y_test, model)
  plt.plot(fpr, tpr, label='{} auc - {}'.format(model, auc))
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.legend()

## Fails for all the above models 
# Try PCA with dimensionality 10 

pca = PCA(n_components=10, svd_solver='arpack')
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print(pca.explained_variance_ratio_)

fpr, tpr, auc, f1, precision, recall = train_models(x_train_pca, y_train, x_test_pca, y_test, '')


