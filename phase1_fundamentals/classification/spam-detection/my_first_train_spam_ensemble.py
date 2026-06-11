# DEFINE PROJECT PATH #################################################
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

nb_model = MultinomialNB()
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)
svc_model = SVC(kernel='linear')

nb_model.fit(x_train_vec,y_train)
rf_model.fit(x_train_vec,y_train)
lr_model.fit(x_train_vec,y_train)
svc_model.fit(x_train_vec,y_train)

# PREDICT #############################################################
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
nb_y_pred = nb_model.predict(x_test_vec)
nb_accuracy = accuracy_score(y_test, nb_y_pred)
print(f"\nModel Accuracy: {nb_accuracy*100:.2f}%")

rf_y_pred = rf_model.predict(x_test_vec)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"\nModel Accuracy: {rf_accuracy*100:.2f}%")

lr_y_pred = lr_model.predict(x_test_vec)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"\nModel Accuracy: {lr_accuracy*100:.2f}%")

svc_y_pred = svc_model.predict(x_test_vec)
svc_accuracy = accuracy_score(y_test, svc_y_pred)
print(f"\nModel Accuracy: {svc_accuracy*100:.2f}%")



# SAVE TRAINING MODEL ##################################################
import pickle

vectorizer_file = MODELS_PATH / 'my_first_spam_ensemble_vectorizer.pkl'

# with open(model_file, 'wb') as f:
#     pickle.dump(model,f) 
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f) # pickle - python built in serialization module, convert python object into a  byte stream(serialization)

models_to_save = {
    'my_naive_bayes': {
        'model': nb_model,
        'test_accuracy' : nb_accuracy
    },
    'my_random_forest':{
        'model': rf_model,
        'test_accuracy': rf_accuracy
    },
    'my_logistic_regression':{
        'model': lr_model,
        'test_accuracy': lr_accuracy
    },
    'my_svc':{
        'model': svc_model,
        'test_accuracy': svc_accuracy
    }
}
import joblib
for model_name, model_data in models_to_save.items():
    model_path = MODELS_PATH /f'{model_name}.pkl'
    # with open (model_path, 'wb') as f:
    #     pickle.dump(model_data,f)
    joblib.dump(model_data, model_path) # better approach
    





