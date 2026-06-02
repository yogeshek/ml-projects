🎯 What You Should ADD/LEARN:
1. Cross-Validation (Instead of just one train-test split)

from sklearn.model_selection import cross_val_score

# Add after line 91 (after model creation):
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
Why? Single train-test split can be lucky/unlucky. CV gives more reliable performance estimate.

2. Compare Multiple Models (Not just Naive Bayes)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC()
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
Why? Learn that different algorithms perform differently on same data.

3. Precision/Recall Analysis (Understand the tradeoffs)
Your current code shows classification report, but you should interpret it:

Precision: Of all messages you flagged as spam, how many were actually spam?
Recall: Of all actual spam messages, how many did you catch?
For spam detection: High recall is crucial (don't miss spam), but precision matters too (don't block legitimate messages)
Add this interpretation:


from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nPrecision: {precision*100:.2f}% - If we flag as spam, we're correct this often")
print(f"Recall: {recall*100:.2f}% - We catch this % of all spam messages")
print(f"\nMissed spam messages: {((1-recall)*y_test.sum()):.0f} out of {y_test.sum()}")
4. Feature Importance (What words indicate spam?)

# Add after model training:
feature_names = vectorizer.get_feature_names_out()
# Get coefficients (for Naive Bayes, use log probabilities)
log_probs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
top_spam_words_idx = log_probs.argsort()[-20:][::-1]

print("\nTop 20 words that indicate SPAM:")
for idx in top_spam_words_idx:
    print(f"  - {feature_names[idx]}")
Why? Understand WHAT the model learned. Does it make sense?

5. Hyperparameter Tuning (Optimize performance)

from sklearn.model_selection import GridSearchCV

# Try different TF-IDF settings
vectorizer_params = {
    'max_features': [1000, 2000, 3000],
    'ngram_range': [(1,1), (1,2)],  # unigrams vs unigrams+bigrams
    'min_df': [1, 2, 5]
}

# Or tune the model itself
model_params = {
    'alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter for Naive Bayes
}

grid = GridSearchCV(MultinomialNB(), model_params, cv=5, scoring='f1')
grid.fit(X_train_vec, y_train)
print(f"\nBest parameters: {grid.best_params_}")
print(f"Best F1 score: {grid.best_score_:.4f}")
Why? Learn that hyperparameters matter and how to tune them.

6. ROC Curve & AUC (Visualize performance)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability predictions
y_proba = model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Spam Detection')
plt.legend()
plt.savefig(MODELS_PATH / 'roc_curve.png')
print(f"\nAUC Score: {auc:.4f}")
Why? Understand model discrimination ability across different thresholds.

7. Error Analysis (Learn from mistakes)

# Find misclassified examples
errors = X_test[y_test != y_pred]
error_labels = y_test[y_test != y_pred]
error_preds = y_pred[y_test != y_pred]

print("\n=== Error Analysis ===")
print(f"\nTotal errors: {len(errors)}")
print("\nFalse Positives (Ham classified as Spam):")
fp = errors[error_labels == 0]
for i, msg in enumerate(fp[:3]):
    print(f"{i+1}. {msg[:100]}...")

print("\nFalse Negatives (Spam classified as Ham):")
fn = errors[error_labels == 1]
for i, msg in enumerate(fn[:3]):
    print(f"{i+1}. {msg[:100]}...")
Why? Understand where your model fails and why.

8. Advanced Text Preprocessing (Beyond basic cleaning)

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def advanced_clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords manually or add stemming
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)
Why? Learn impact of different preprocessing techniques.

9. Class Imbalance Handling (If dataset is imbalanced)

from sklearn.utils.class_weight import compute_class_weight

# Check if imbalanced
print(f"Class distribution: {y_train.value_counts()}")

# If imbalanced, use class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model = MultinomialNB(class_prior=class_weights)
10. Visualizations (Make insights clear)

import seaborn as sns

# Confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig(MODELS_PATH / 'confusion_matrix.png')

# Message length distribution
df['msg_length'] = df['message'].str.len()
plt.figure(figsize=(10, 5))
df.groupby('label')['msg_length'].hist(alpha=0.5, bins=50, legend=True)
plt.xlabel('Message Length')
plt.title('Spam vs Ham Message Length')
📚 Key Concepts to Learn:
Stratified splitting - You already do this! (line 54)
fit vs transform vs fit_transform - You asked about this (line 75-76)
fit: Learn patterns from training data
transform: Apply learned patterns to new data
fit_transform: Do both at once (only on training data!)
Why max_features matters - More features = more memory, potential overfitting
N-grams - Currently using unigrams only, try bigrams ("free money" vs "free" + "money")
Bias-Variance tradeoff - Simple models (underfitting) vs complex models (overfitting)
🎯 Immediate Action Items:
Would you like me to:

Add cross-validation to your current code?
Compare multiple models (NB vs Logistic Regression vs SVM)?
Add ROC curve visualization?
Implement error analysis to see where model fails?
All of the above - create an enhanced version?
Which would you like to tackle first?