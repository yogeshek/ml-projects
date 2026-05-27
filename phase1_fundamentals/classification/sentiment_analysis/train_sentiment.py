import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'IMDB Dataset.csv'
MODELS_PATH = PROJECT_ROOT / 'models'

#1. Reading Data
df=pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = ['review','sentiment']
# print(df.head())


#2 process data
#convert label to binary

df['sentiment']=df['sentiment'].map({'positive':0, 'negative':1})

#clean test
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join(text.split())
    return text
df['review']=df['review'].apply(clean_text)
print(df.head())

#split data , separate x and target y
x=df['review']
y=df['sentiment']

from sklearn.model_selection import train_test_split
#slpit into train and test sets [80:20]
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y    
)

print(f"training set size: {len(x_train)}")
print(f"testing set size: {len(x_test)}")
print(f"negative in training: {sum(y_train)/len(y_train)*100:.1f} ")
print(f"negative in testing: {sum(y_test)/len(y_test)*100:.1f}")

#4 vectorize the text
#convert the review into numerical features that the ML model can understand
#tf-idf term frequency and inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features= 10000, #10000 most important words
    stop_words='english', # remove common word is, the etc
    lowercase=True
)

#vectorize the train and test data- assign tf*idf scores to each term
#and transfrom them to matrix

x_train_vec = vectorizer.fit_transform(x_train)
# print(x_train_vec)
# vocab= vectorizer.get_feature_names_out()
# print(vocab[23])
x_test_vec = vectorizer.transform(x_test)

print(f"vectorized training size: {x_train_vec.shape}")
print(f"vectorized testing size: {x_test_vec.shape}")
print(f"number of features: {len(vectorizer.get_feature_names_out())}")

#5 train the model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_vec,y_train)

#6 Evaluate the model
#test the trined model on the test 

y_pred = model.predict(x_test_vec)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test,y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

#detailed classification report
print(classification_report(y_test,y_pred,target_names=['positive','negative']))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#save the training model
import pickle
MODELS_PATH.mkdir(exist_ok=True)
model_file = MODELS_PATH / 'sentiment_classifier.pkl'
vectorizer_file = MODELS_PATH / 'sentiment_vectorizer.pkl'

with open(model_file,'wb') as f:
    pickle.dump(model,f)
with open(vectorizer_file,'wb') as f:
    pickle.dump(vectorizer,f)

print("\nModel and Vectorizer saved successfully!")
print(f"- {model_file}")
print(f"- {vectorizer_file}")

    


    

