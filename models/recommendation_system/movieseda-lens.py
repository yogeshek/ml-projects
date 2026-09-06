import pandas as pd
import numpy as np

import urllib.request
import os
import zipfile
import ssl

# url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
# zip_path = 'ml-100k.zip'

# if not os.path.exists(zip_path):
#     context = ssl._create_unverified_context()
#     response = urllib.request.urlopen(url, context=context)
#     with open(zip_path, 'wb') as f:
#         f.write(response.read())

# with zipfile.ZipFile(zip_path, 'r') as z:
#     z.extractall('.')

df_data = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                  names=['user_id', 'item_id', 'rating', 'timestamp'])
# print(df_data.head())

df_user= pd.read_csv('ml-100k/u.user', sep='\t', header=None,
                names=['user id' , 'age',  'gender',  'occupation',  'zip code'])
# print(df_user.head())

df_item = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None)
# print(df_item.head())

################## find the weight as per the site
# W = Rv +Cm  / v+ m 
# W =weighted rating
# R = Average for the movie as a number form 0 to 10
# v = number of votes for the movie
# m = minimum vote required to be listed in top 250 (CURRENTLY 3000)
# C = the mean vote accross the whole report

# lets to some EDA
# How many users / movies are there ?
# R : Find the number of users rated any single movie
# v : Find the average rating of a movie
# C : Find the average overall rating

# print(df_data.head())
# print(df_data.shape)

print(df_data['item_id'].nunique())
print(df_data['user_id'].nunique())

v= (df_data.groupby(['item_id'])['rating'].mean())
# print(v)

R = (df_data.groupby(['item_id'])['rating'].count())
# print(R)

C=df_data['rating'].mean()
# print(C)

# orininally m=3000 as per the site, from the data them max votes=583 for a movie, so we keep our threshold to 50
# m = np.where(R >= 40)[0]
# print(len(m))
m = 50

# print(R.max())

#find the weignt
W =( R*v +C*m ) /( v+ m)
# print(W)


weighted_rating = pd.DataFrame()
weighted_rating['item_id']=W.index
weighted_rating['W']=W.values

# print(weighted_rating.head(5))


imdb = weighted_rating.merge(df_item, left_on='item_id',right_on=0)
# print(imdb)

print(imdb.sort_values(by = "W", ascending=False).head(5))
