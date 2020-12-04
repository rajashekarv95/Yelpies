from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import string
import spacy
import nltk
from textblob import TextBlob
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import requests
import json
import time
import os

def get_redirect(id, verbose = False):
  response = requests.get('https://www.yelp.com/biz/{}'.format(id))
  if response.history:
      if verbose:
        print("Request was redirected")

      resp = response.history[-1]
      if verbose:
        print(type(resp))
        print(resp.status_code, resp.url)
        print("Final destination:")
        print(response.status_code, response.url)
  else:
      if verbose:
        print("Request was not redirected")
  return response.url



mapping = pd.read_csv('mapping.csv')

# Specify the start and end of
start = 0
end = 500

mapping = mapping[start:end]


df = pd.read_csv('scraped_parallel.csv')

if len(df) != 0:
    last_bid = df.iloc[-1][0]
else:
    last_bid = ''

df = df[df.business_id != last_bid]

ids_scraped = df['business_id'].unique()
err = []
t = time.time()

for row in mapping.itertuples():
    i = 0
    print(row)
    if row.business_id in ids_scraped:
        print('Already scraped')
        continue
    while True:
        # get webpage
        # url = 'https://www.yelp.com/biz/bl3-rGjqjaJa_nkW4aMlIg&start={}&sort_by=date_desc'.format(i)
        url = get_redirect(row.business_id, verbose = False)
        query = '?start={}&sort_by=date_desc'.format(i)

        url += query
        # print(url)
        flag = False
        try:
            ourUrl = urllib.request.urlopen(url)
        except:
            flag = True
            err.append(row.business_id)
            pass
        if flag:
            break
        soup = BeautifulSoup(ourUrl,'html.parser')
        rows = soup.find_all('div',{'class': 'main-content-wrap main-content-wrap--full'})
        # print(rows[0].text[6:-4])
        rows = json.loads(rows[0].text[6:-4])

        #Increment the page number specifier
        i += 20

        #Break if no rows found
        if not rows['bizDetailsPageProps']['reviewFeedQueryProps']['reviews']:
            break
        df_temp = pd.json_normalize(rows['bizDetailsPageProps']['reviewFeedQueryProps']['reviews'])

        required_columns = ['localizedDate', 'comment.text', 'rating']
        col_names = ['date', 'review', 'rating']

        df_req = df_temp[required_columns]
        df_req.columns = col_names
        
        df_req['business_id'] = row.business_id

        df = df.append(df_req, ignore_index=True)
        time.sleep(2)
        df.to_csv('scraped_parallel.csv',index=False)
    print(df.shape, time.time() - t)



df.to_csv('scraped_parallel.csv',index=False)

