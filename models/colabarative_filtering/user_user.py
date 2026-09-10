import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 — Load data
columns = ['user id' , 'item id' , 'rating' , 'timestamp']
data = pd.read_csv('../content_based_filtering/ml-100k/u.data', sep='\t',encoding='latin-1', header=None,
                   names = columns )

columns = ['item_id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western' ]

item = pd.read_csv('../content_based_filtering/ml-100k/u.item', sep ='|', encoding='latin-1', header=None,
                   names = columns )

# Step 2 — Create item-user matrix (pivot table)
# in case of item-item we have items as row and user as column, here we need users as row instead

user_pivot = data.pivot_table(index='user id',columns='item id', values='rating', fill_value=0)
print(user_pivot)

# Step 3 — Compute cosine similarity between items

cosine_sim_matrix = cosine_similarity(user_pivot)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_pivot.index, columns=user_pivot.index)
print(cosine_sim_df)

# Step 4 — For a given user, separate rated vs unrated movies
target_user = 8

rated = user_pivot.loc[target_user][user_pivot.loc[target_user]>0]
unrated = user_pivot.loc[target_user][user_pivot.loc[target_user]==0]
print(rated) # userid


movie_id= unrated.index[0]
item_rating = user_pivot[movie_id]
user_rated = item_rating[item_rating>0] # user id and rating


sim = cosine_sim_df.loc[target_user,user_rated.index]
prediction = (sim * user_rated).sum() / sim.sum()
print('****************')
print(prediction)

score_df = pd.DataFrame(columns =['item_id','score'])
for movie_id in unrated.index:
    item_rating = user_pivot[movie_id]
    user_rated = item_rating[item_rating>0]

    sim = cosine_sim_df.loc[target_user, user_rated.index]
    prediction = (sim * user_rated).sum() / sim.sum()

    score_df.loc[movie_id, 'item_id'] = movie_id
    score_df.loc[movie_id, 'score'] = prediction
score_df = score_df.sort_values(by='score',ascending=False)
print(score_df.head(50))


prediction_df=score_df.merge(item[['item_id','movie title']], on='item_id',how='left')
print(prediction_df) 