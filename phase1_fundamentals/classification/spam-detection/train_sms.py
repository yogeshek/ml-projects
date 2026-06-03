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
    # Multinomial Naive Bayes (MultinomialNB)
    # Uses Bayes' theorem to classify text.
    # Normally expects word counts, but can also use TF-IDF weights
    # (treated as weighted/fractional counts).

    # Example training data:
    # 1. "excellent movie good to watch" -> Positive
    # 2. "poor movie don't watch"        -> Negative

    # Step 1: Build vocabulary
    # [excellent, movie, good, to, watch, poor, don't]
    # Vocabulary size V = 7

    # Step 2: Count words per class
    # Positive: [1,1,1,1,1,0,0], Npos = 5
    # Negative: [0,1,0,0,1,1,1], Nneg = 4

    # Step 3: Calculate conditional probabilities using
    # Laplace smoothing (alpha = 1)

    # Positive denominator = Npos + alpha*V = 5 + 7 = 12
    # P(excellent|Positive) = (1+1)/12 = 2/12
    # P(movie|Positive)     = (1+1)/12 = 2/12
    # P(good|Positive)      = (1+1)/12 = 2/12
    # P(to|Positive)        = (1+1)/12 = 2/12
    # P(watch|Positive)     = (1+1)/12 = 2/12
    # P(poor|Positive)      = (0+1)/12 = 1/12
    # P(don't|Positive)     = (0+1)/12 = 1/12

    # Negative denominator = Nneg + alpha*V = 4 + 7 = 11
    # P(excellent|Negative) = (0+1)/11 = 1/11
    # P(movie|Negative)     = (1+1)/11 = 2/11
    # P(good|Negative)      = (0+1)/11 = 1/11
    # P(to|Negative)        = (0+1)/11 = 1/11
    # P(watch|Negative)     = (1+1)/11 = 2/11
    # P(poor|Negative)      = (1+1)/11 = 2/11
    # P(don't|Negative)     = (1+1)/11 = 2/11

    # Step 4: Classify new sentence: "excellent movie watch"

    # Score(Positive)
    # = P(Positive) * P(excellent|Positive)
    #               * P(movie|Positive)
    #               * P(watch|Positive)

    # Score(Negative)
    # = P(Negative) * P(excellent|Negative)
    #               * P(movie|Negative)
    #               * P(watch|Negative)

    # Predict the class with the higher score.
        # Scorepos​=0.5×P(excellent∣Positive)×P(movie∣Positive)×P(watch∣Positive) = 0.5×2/12​×2/12​×2/12 = 0.00231
        # ScoreNeg​=0.5×P(excellent∣Negative)×P(movie∣Negative)×P(watch∣Negative) = 0.5×1/11​×2/11​×2/11 = 0.00150
        # Prediction = Positive as​ 0.00231 > 0.00150      
    
    # In practice, scikit-learn uses log probabilities for numerical stability.    
    
rf_model = RandomForestClassifier()
    # Ensemble learning algorithm that combine multiple decision tree.
    # Each tree is trained on a random subset of rows and features
    # Final decision is made by majority voting among all trees
    
    # Example Training Data
    # ---------------------------------------------------
    # Review                          Label
    # ---------------------------------------------------
    # "excellent movie good watch"    Positive
    # "poor movie don't watch"        Negative
    # "good story excellent acting"   Positive
    # "boring movie poor acting"      Negative
    
    # Step 1: Convert text into numeric features
    # (e.g. TF-IDF)
    #
    #               excellent movie good watch poor acting boring
    # Review1  ->      1.2      0.3   0.8  0.4   0.0   0.0   0.0
    # Review2  ->      0.0      0.3   0.0  0.4   1.1   0.0   0.0
    # Review3  ->      1.0      0.0   0.9  0.0   0.0   1.2   0.0
    # Review4  ->      0.0      0.5   0.0  0.0   1.0   1.0   1.3
    
    # Step 2: Create multiple Decision Trees
    #        
    # Tree 1:
        # - Trained on random rows
        # random rows - Bootstrap Sampling with replacement
        # Eg Original rows 1, 2, 3, 4, Random sample for tree1 1, 1, 3, 4
        
        # - Uses random subset of features
        # random subset of features
        # suppose we have 8 [feauture excellent movie good watch poor acting boring story], A normal Decision Tree considers all 8 features at every split.
        # Random Forest does not use all 8
        # Random Forest choose lets say 3 feature-random subset(excellent,poor,story) and only one of the 3 feature(best- based on entropy) is allowed to split the root node.
        # At the next node- let say 3 differnet feaures - random subset(acting,watch,poor) are slected and 1 best among the 3 is allowed to split it
        # subset should be root(total features),2 or 3 when total is 8
    # Tree 2: Repeat the same as tree1
    # Tree 3: same as above
    
    # Step 3: Prediction
    # predict "excellent movie watch"
    # Tree 1 -> Positive
    # Tree 2 -> Positive
    # Tree 3 -> Negative
    # Tree 4 -> Positive
    # Tree 5 -> Positive
    
    #votes : Positive = 4
    #        Negative = 1
    #final prediction: Positive         
          

lr_model = LogisticRegression(max_iter=1000)
    # For LogisticRegression the algorithm is very different from NB(learns word probabiliyt) and RF(learns decision rules)
    # linear classification algorithm that learns  a weight for each feaure
    # positive word get positive wieghts and negative word get negative weights
    # the weighted sum is passed through the sigmoid function to obtain the probability

    # Example training data:
    # 1. "excellent movie good watch" -> Positive (1)
    # 2. "poor movie don't watch"     -> Negative (0)
    
    # Step 1: Build vocabulary
    # [excellent, movie, good, watch, poor, don't]
    
    # Step 2: Convert text to TF-IDF features
    #
    # Review1 -> [1.2, 0.3, 0.8, 0.4, 0.0, 0.0]
    # Review2 -> [0.0, 0.3, 0.0, 0.4, 1.1, 1.0]
    
    # Step 3 : Learn weight from training data
    # for "excellent movie good watch"
    # model learns z=b+w1​x1​+w2​x2​+⋯+wn​xn​
    
    # starting with w=0 b=0, then z=0
    # P(y=1)=1 / 1+e−z as z=0, ​P(y=1)=0.5 but the actual label is positive=1, predicted = 0.5
    
    # there fore weight get updtaed Wnew​=Wold​−η×gradient
    # η= Step size and gradient = =(y^​−y)xj​. Xexcellent​=1, So (y^−y)x=(0.5−1)×1=−0.5, gradient=-0.5
    # Wnew=0−0.1(−0.5) = 0.05 
    # Example learned weights:
    #
    # excellent = +2.5
    # good      = +1.8
    # movie     = +0.1
    # watch     = +0.2
    # poor      = -2.2
    # don't     = -1.9
    #
    # Positive words get positive weights.
    # Negative words get negative weights.

    # Step 4: Predict new sentence
    #
    # "excellent movie watch"
    #
    # TF-IDF:
    # excellent = 1.3
    # movie     = 0.2
    # watch     = 0.4

    # Linear score (z):
    #
    # z = intercept + 1.3*(2.5) + 0.2*(0.1) + 0.4*(0.2)
    # Assume intercept = -1
    # z = -1 + 3.25 + 0.02 + 0.08
    # z = 2.35
    
    # Step 5: Convert score to probability using sigmoid, z=2.35
    # P(Positive) = 1 / (1 + e^(-z)) =0.913
    
    # Step 6: Classification
    #
    # If probability >= 0.5:
    #     Positive
    # Else:
    #     Negative
    # 0.913 > 0.5 Prediction = Positive
    
 

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