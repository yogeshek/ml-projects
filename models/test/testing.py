from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document',
    'This document is the second document'
]

Vectorizer = CountVectorizer(lowercase=True, ngram_range=(2,2))
X = Vectorizer.fit_transform(corpus)
print(Vectorizer.vocabulary_)

X_array = X.toarray()
print(X_array)

# import pandas as pd
# import numpy as np

# df = pd.DataFrame({
#     "Name":["Amit","Bhuvana","chetan","Divya","Ekta"],
#     "Maths":[78, None, 92, 85, 60],
#     "Science":[88, 90, None, None, 75],
#     "Section":["A", "A", "B", "B", "A"]
# })

# df["total"]=df[["Maths","Science"]].fillna(0).sum(axis=1)
# # print(df)
# # df["total"]=df["Maths"]+df["Science"]
# # print(df)
# df["total"]=df.Maths.add(df.Science,fill_value=0)
# # print(df)
# df = df.assign(total=lambda x:x["Maths"].add(x["Science"],fill_value=0))
# print(df)