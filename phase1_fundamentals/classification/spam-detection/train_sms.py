from pathlib import Path
import pandas as pd


#### 1. Get the project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' /'spam_sms.csv'
MODELS_PATH = PROJECT_ROOT / 'models'

#### 2. Load data
df = pd.read_csv(DATA_PATH, encoding='latin-1')
# print(df.head())
df.columns = ['Class', 'Message']
# print(df.columns)
# print(f"df shape : {df.shape}")

#### 3. Pre process data
#convert label to binary
df['Class']=df['Class'].map({'ham':0, 'spam':1}) # mapping to binary and assigning back to the original column/series

#cleand the text(Message)
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join(text.split())
    return text
df['Message'] = df['Message'].apply(clean_text)
# print(df.head())

# text = "Hello WORLD 123! @#$"
# clean = clean_text(text)
# print(clean)

####### 4 . split the data into feature x and target y
from sklearn.model_selection import train_test_split
x = df['Message']
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size= 0.2,
    random_state= 42,
    stratify=y    
)
# print(f"training set: {len(x_train)}")
# print(f"testing set: {len(x_test)}")
# print(sum(y_train)/len(x_train)*100)
# print(sum(y_test)/len(x_test)*100)
# print(sum(y)/len(x)*100)

######## 5 .Vectorize text
#convert text messages into numerical features
#tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features= 1000,
    stop_words= 'english',
    lowercase= True
)

#TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
#IDF(t) = log( (Total number of documents) / (Number of documents containing term t) ) + 1
# TF-IDF = TF*IDF
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test) 

print(f"\nvectorized training shape: {x_train_vec.shape}")
print(f"vectorized testing shpe: {x_test_vec.shape}")
print(f"number of features: {len(vectorizer.get_feature_names_out())}")  

####### 6 Train model
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
nb_model = MultinomialNB()
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)
svc_model = SVC(kernel='linear')

nb_model.fit(x_train_vec, y_train)
rf_model.fit(x_train_vec, y_train)
lr_model.fit(x_train_vec, y_train)
svc_model.fit(x_train_vec, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

nb_y_pred =nb_model.predict(x_test_vec)
nb_accuracy =  accuracy_score(y_test, nb_y_pred)
nb_precision = precision_score(y_test, nb_y_pred)
nb_recall = recall_score(y_test, nb_y_pred)
# print(f"nb_accuracy :{nb_accuracy*100:.1f}")

rf_y_pred =rf_model.predict(x_test_vec)
rf_accuracy =  accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
# print(f"rf_accuracy :{rf_accuracy*100:.1f}")

lr_y_pred =lr_model.predict(x_test_vec)
lr_accuracy =  accuracy_score(y_test, lr_y_pred)
lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
# print(f"lr_accuracy :{lr_accuracy*100:.1f}")

svc_y_pred =svc_model.predict(x_test_vec)
svc_accuracy =  accuracy_score(y_test, svc_y_pred)
svc_precision = precision_score(y_test, svc_y_pred)
svc_recall = recall_score(y_test, svc_y_pred)
print(f"nb_accuracy = {nb_accuracy*100:.1f} : rf_accuracy = {rf_accuracy*100:.1f} : lr_accuracy = {lr_accuracy*100:.1f}, svc_accuracy = {svc_accuracy*100:.1f} ")
print(f"nb_precision = {nb_precision*100:.1f} : rf_precision = {rf_precision*100:.1f} : lr_precision = {lr_precision*100:.1f}, svc_precision = {svc_precision*100:.1f} ")
print(f"nb_recall = {nb_recall*100:.1f} : rf_recall = {rf_recall*100:.1f} : lr_recall = {lr_recall*100:.1f}, svc_recall = {svc_recall*100:.1f} ")

print("\nClassification Report:")
print(classification_report(y_test,nb_y_pred,target_names=['ham','spam']))
    
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

auc = roc_auc_score(y_test, nb_y_pred)
print(f"AUC: {auc:.3f}")

#plot roc curve
fpr, tpr, _ = roc_curve(y_test, nb_y_pred)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


#confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, nb_y_pred)
print(cm)
print("\n[[True Ham, False Spam]")
print("[False Ham, True Spam]]")