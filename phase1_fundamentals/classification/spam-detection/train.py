import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'spam.csv'
MODELS_PATH = PROJECT_ROOT / 'models'

df =pd.read_csv(DATA_PATH, encoding='latin-1')
df =df[['v1','v2']]
df.columns = ['label','message']

print(f"data shape: {df.shape}")
print(f"\nfirst 5 rwos:")
print(df.head())
print(f"\nClass distribution:")
print(df['label'].value_counts())

#pre process data
#convert labels to binary

df['label']= df['label'].map({'ham':0, 'spam': 1})

#clean the text
import re

def clean_text(text):
    text = text.lower() #convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]','',text) #remove spl char
    text = ' '.join(text.split()) # remove extra white space
    return text
df['message'] = df['message'].apply(clean_text)

print("\nAfter preprocessing:")
print(df.head())

#split data
#seperate features (x) and target (y)
X = df['message']
Y = df['label']

#split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"Spam in training: {y_train.sum()}({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"Spam in tersing: {y_test.sum()}({y_test.sum()/len(y_test)*100:.1f}%)")

#4 vectorize text
#converts text message into numerical features that the ML model can understand
#tf-idf term frequency-inverse document frequency

vectorizer = TfidfVectorizer(
    max_features=2000, #keep top 3000 most important words
    stop_words='english', #Remove common words like the, is , and
    lowercase=True
)

#fit on training data and transform both train and test
#ASK TO EXPLAIN IN DETAILS

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nvectorized training shape: {X_train_vec.shape}")
print(f"vectorized testing shpe: {X_test_vec.shape}")
print(f"number of features: {len(vectorizer.get_feature_names_out())}")

#EACH MESSAGE IS Now represneted by 300 numbers

#5 train model
#train a naive Bayes classifier on the training data
#naive bayes works very well for the text classification

#create and train the model
model = MultinomialNB()
print(f"Training the model ...")
model.fit(X_train_vec,y_train)
print(f"Training complete")    

#6.Evaluate model
#Test the trained model on unseen data and calculate performance metrics
#check accuracy pression recall

y_pred =model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

#detailed classification report
print("\nClassification Report:")
print(classification_report(y_test,y_pred,target_names=['Ham','Spam']))

#confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n[[True Ham, False Spam]")
print("[False Ham, True Spam]]")

#save the training model
MODELS_PATH.mkdir(exist_ok=True)
model_file = MODELS_PATH / 'spam_classifier.pkl'
vectorizer_file = MODELS_PATH / 'vectorizer.pkl'

with open(model_file,'wb') as f:
    pickle.dump(model,f)
#save the vectorizer
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and Vectorizer saved successfully!")
print("Files saved:")
print(f"- {model_file}")
print(f"- {vectorizer_file}")

    
    
