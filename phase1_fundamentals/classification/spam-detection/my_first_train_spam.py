# DEFINE PROJECT PATH ##########################################
from pathlib import Path

#Get prject root directory
PROJECT_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_PATH/'data'/'spam.csv'
MODELS_PATH = PROJECT_PATH/'models'

# READ AND LOAD THE DATA FILE #########################################
import pandas as pd
df =pd.read_csv(DATA_PATH,encoding='latin-1')
df=df[['v1','v2']]    # select only first 2 columns
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham':0, 'spam':1}) #convert label to binary

# CLEAN DATA ##########################################################
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text) # r-raw string prefix, \s white space characters, [^a-zA-Z]- any char thats not a letter
    text = ' '.join(text.split())# text.split() geneate a word list and ' '.join the words with single space
    return text
df['message'] =df['message'].apply(clean_text)

# SPLIT AND TRAIN THE DATA ##############################################
from sklearn.model_selection import train_test_split

x= df['message']
y= df['label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# VECTORIZE THE DATA ################################################
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features= 2000,
    stop_words='english',
    lowercase=True
)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# TRAIN THE MODEL ####################################################
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_vec,y_train)

# PREDICT #########################################################
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")








