import pandas as pd
import numpy as np
import os
import ssl
import urllib.request
import zipfile


url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
zip_path = 'ml-100k.zip'

if not os.path.exists(zip_path):
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(url=url, context=context)
    with open(zip_path, 'wb') as f:
        f.write(response.read())
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('.')

data_df = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                      names = ['user id' , 'item id' , 'rating' , 'timestamp'])
# print(data_df.head())

columns = ['movie id' , 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western' ]
item_df = pd.read_csv('ml-100k/u.item', encoding='latin-1', sep='|', header=None,names=columns
                      )
# print(item_df.head())

# find the content based similarity amonng the items. the similariy using cosine- similarity
# cos(teta) = A*B / ||A||*||B||
#||A||= A * A  and ||B||= B * B

#sample manual 
V1= np.array(([1,2,3]))
V2 = np.array(([-1,-2,-3]))
V3 = np.array(([2,7,9]))

nm = np.dot(V1,V2.T) # OR
nm = np.matmul(V1,V2)
dm = np.sqrt(np.sum(V1**2)) * np.sqrt(np.sum(V2**2))
cossine = nm /dm
# print(f"V1, V2:{cossine}")

nm = np.dot(V1,V3.T)
dm = np.sqrt(np.sum(V1**2)) * np.sqrt(np.sum(V3**2))
cossine = nm /dm
# print(f"V1, V3:{cossine}")

nm = np.dot(V2,V3.T)
dm = np.sqrt(np.sum(V2**2)) * np.sqrt(np.sum(V3**2))
cossine = nm /dm
# print(f"V2, V3:{cossine}")

# or
cossine_similarity = np.dot(V2,V3.T) / np.sqrt(np.sum(V2**2)) * np.sqrt(np.sum(V3**2))
# print(f"V2, V3:{cossine}")


## using cosine_similarity method

from sklearn.metrics.pairwise import cosine_similarity
co_v1_v2 = cosine_similarity([V1], [V2])
# print(co_v1_v2)

##################################################################333
# print(item_df.head())
# will take genre to cind the cosine similarity

movie_content = item_df.loc[:,'unknown':]
# print(movie_content)
# cosine_similarity= cosine_similarity(movie_content,movie_content)
# sim = pd.DataFrame(cosine_similarity)
# print(sim)

movie_content = item_df.loc[:,'unknown':]
cos_sim = cosine_similarity(movie_content, movie_content)
sim = pd.DataFrame(cos_sim)

index = 89 # item selected to find similar items
k = 10 # top 10 items

indices = sim.loc[:,index].drop(index).sort_values(ascending=False).head(k).index
print(item_df.iloc[index]['movie title'])
print(item_df.iloc[indices]['movie title'])

# creating a function to do the above

def content_based(df, column_idx, start_column, end_column):
    content = df.loc[:,start_column:end_column]
    array = cosine_similarity(content, content)
    sim = pd.DataFrame(array)
    # print(sim)
    indices = sim[column_idx].drop(column_idx).sort_values(ascending=False).head(5).index
    # print(indices)
    print(f"wathced movie:\n {df.iloc[column_idx]['movie title']}")
    return df.iloc[indices]
    

similarity = content_based(item_df, column_idx=3, start_column='unknown', end_column='War')
print(f"Reccomended movie:\n {similarity['movie title']}")    
   
   
