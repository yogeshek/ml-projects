import pandas as pd
import numpy as np

################ Step 1 — Load data

columns = ['user id' , 'item id' , 'rating' , 'timestamp']
data = pd.read_csv('../content_based_filtering/ml-100k/u.data', sep='\t', header=None,
                   names=columns)
print(data)


columns = ['movie id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western' ]
items = pd.read_csv('../content_based_filtering/ml-100k/u.item', sep='|', header=None, encoding='latin-1',
                   names=columns)
# print(items)

############# Step 2 — Create user-item matrix (pivot table)

data_pivot = pd.pivot_table(data, index='item id', columns='user id', values='rating', fill_value=0)
# print(data_pivot)

########### Step 3 — Compute cosine similarity between items

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim =cosine_similarity(data_pivot, data_pivot)

sim = pd.DataFrame(cosine_sim, index=data_pivot.index, columns=data_pivot.index) # replace the row index with item_id
# print(sim)

######### Step 4 — For a given user, separate rated vs unrated movies

user_id=8
rated_movies = data_pivot[data_pivot.loc[:,user_id]>0][user_id]
# print(rated_movies)
unrated_movies = data_pivot[data_pivot.loc[:,user_id]==0][user_id]
# print(unrated_movies)

######### Step 5 — Predict ratings for each unrated movie
# simscore * ratings / sum(sumscore)
#For each unrated movie, get its similarity with the user's rated movies

# dot producted of unrated move simscore and rated movies ratings

unrated_movie_idx = unrated_movies.index[0] #unrated movie first id
num1 = sim.loc[unrated_movie_idx,rated_movies.index] # simscore unrated vs rated
num2 = rated_movies # rated items with rating
den = np.sum(num1) # total of simscore

prediction = num1.dot(num2) / den
print(prediction)

score_df = pd.DataFrame(columns =['item_id','score'])
# prediction = {}
for i in unrated_movies.index:
    num1 = sim.loc[i,rated_movies.index]
    num2 = rated_movies
    den = np.sum(num1)
    #  prediction[i] = num1.dot(num2) / den
    prediction = num1.dot(num2) / den
    
    score_df.loc[i, 'item_id']= i
    score_df.loc[i, 'score'] = prediction
score_df = score_df.sort_values(by='score', ascending=False)[:10]
print(score_df.head(10))
# prediction=pd.Series(prediction).sort_values(ascending=False)
# print(prediction)

indices = score_df.index
print(indices)


# recommended = score_df.merge(items, left_on='item_id', right_on='movie id')
# print(recommended[['item_id', 'score', 'movie title']].head(100))
print(items.loc[indices-1])

