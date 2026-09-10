import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

columns = ['movie id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western' ]
item = pd.read_csv('../content_based_filtering/ml-100k/u.item',delimiter='|',encoding='latin-1',header=None,
                   names=columns)
# print(item.head(5))


columns = ['user id' , 'item id' , 'rating' , 'timestamp']
data = pd.read_csv('../content_based_filtering/ml-100k/u.data',delimiter='\t',encoding='latin-1',header=None,
                   names=columns)

#user pivot table create user - item matrix (index is item_id beacuse it is item-item filtering)

df = data.pivot_table(index='item id',columns='user id', values='rating', fill_value=0)

# before predict the rating seperate rated movies and unrated movies for one of the user says user id=5
user_id=5
rated_movies = df[df.loc[:,user_id]>0][user_id]
unrated_movies = df[df.loc[:,user_id]==0][user_id]
# print(rated_movies)

#find cosine similarity
cosine_sim = cosine_similarity(df,df)
# cosine_sim = pd.DataFrame(cosine_sim)
#instead of row based index we need the real id as the index, so when we refer index 5, it refers the movie/item id= 5,
# with row index when we say 5 it refers tie th movie/item id 6
cosine_sim = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)
# print(cosine_sim)

# lets take one movie id from unrated moves
movie_idx = 3
#get the 
# sim = cosine_sim.loc[movie_idx,rated_movies.index]
# print(sim) # rated moves with cosine score
# print(rated_movies) # rated moves with rating

# ## we do the dot prodcut of sim-rated moves sim score and rated moves with rating

# print(sim.dot(rated_movies)/np.sum(np.abs(sim)))
# print(unrated_movies)





# for all users for all unrated movies
for user_id in df:
    rated_movies = df[df.loc[:,user_id]>0][user_id]
    unrated_movies = df[df.loc[:,user_id]==0][user_id]

    predictions = {}
    
    for i in unrated_movies.index:
        sim = cosine_sim.loc[i,rated_movies.index]
        total_sim = np.sum(np.abs(sim))
        if total_sim > 0:
            rate = sim.dot(rated_movies)/np.sum(np.abs(sim))
            predictions[i] = rate
    predictions= pd.Series(predictions).sort_values(ascending=False)
print(predictions)

