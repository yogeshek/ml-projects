# import zipfile

# with zipfile.ZipFile(r'C:/Users/YogeshEkambaramK/Downloads/archives.zip','r') as z:
#     z.extractall('.')
import pandas as pd
    
data = pd.read_csv('tmdb_5000_movies.csv')
# print(data.head())
# print(data.columns.to_list())
# print(data['overview'])
# print(data.head())

###### will use the overview column
# print(data['overview'].isna().sum())
#fill na with empty
data['overview']=data['overview'].fillna('')
# print(data['overview'].isna().sum())

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
# print(tfidf_matrix.shape)
# print(len(tfidf.vocabulary_))

#just to check the vocabilary
# count = 0
# for i in tfidf.vocabulary_:
#     print(i)
#     count += 1
#     if count ==10:
#         break

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
sim = pd.DataFrame(cosine_sim)
# print(sim.shape)
print(sim)


# one of passing the index and get the top k
idx = 7
indices = sim.loc[idx].drop(idx).sort_values(ascending=False).head(5).index
print(data.iloc[indices][['title']])

# another way is to give the title insted of index directly
# that is again we pass the index only to the sim matrix but first we give the title and get the index then passing the index

#creating the index title series to access the index by giving the title
indices = pd.Series(data.index,data['title']).drop_duplicates()
print(indices)
print("*************")
print(indices['Avatar'])
print("*************")


idx = indices['Avatar']
indices = sim.loc[idx].drop(idx).sort_values(ascending=False).head(5).index
print(data.iloc[indices]['title'])

# lets create a function to do this

def get_recommendations(title, k):
    indices = pd.Series(data.index,index=data['title']).drop_duplicates()
    idx = indices[title]
    top_k =sim[idx].drop(idx).sort_values(ascending=False).head(k).index
    return data.iloc[top_k]['title']

print(get_recommendations('The American',3))
    
