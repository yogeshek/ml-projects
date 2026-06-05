from pathlib import Path
import joblib
import re
from nltk.stem import PorterStemmer

#### 1. Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

#### 2. Load vectorizer and all models
print("Loading models...")
vectorizer = joblib.load(MODELS_PATH / 'vectorizer.pkl')

models = {}
model_files = ['naive_bayes', 'random_forest', 'logistic_regression', 'svc']

for model_name in model_files:
    model_path = MODELS_PATH / f'{model_name}.pkl'
    if model_path.exists():
        models[model_name] = joblib.load(model_path)
        print(f"✅ Loaded {model_name} (Test Acc: {models[model_name]['test_accuracy']*100:.1f}%)")
    else:
        print(f"❌ Model not found: {model_path}")

print(f"\n✅ Loaded {len(models)} models successfully!\n")

#### 3. Text preprocessing function (same as training)
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    text = ' '.join(stemmed_words)
    return text

#### 4. Prediction function for all models
def predict_all_models(message):
    """
    Predict spam/ham using all loaded models
    Returns a dictionary with predictions from each model
    """
    # Preprocess the message
    cleaned_message = clean_text(message)

    # Vectorize the message
    vectorized = vectorizer.transform([cleaned_message])

    # Get predictions from all models
    results = {}
    for model_name, model_data in models.items():
        model = model_data['model']

        # Get prediction (0=ham, 1=spam)
        prediction = model.predict(vectorized)[0]

        # Get probability (if available)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized)[0][1]  # Probability of spam
        else:
            # SVC with linear kernel doesn't have predict_proba by default
            proba = None

        results[model_name] = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'probability': proba,
            'confidence': proba * 100 if proba else None,
            'test_accuracy': model_data['test_accuracy']
        }

    return cleaned_message, results

#### 5. Display results nicely
def display_results(message, cleaned_message, results):
    print("="*70)
    print(f"📧 Original Message: {message}")
    print(f"🧹 Cleaned Message: {cleaned_message}")
    print("="*70)
    print(f"{'Model':<25} {'Prediction':<10} {'Confidence':<15} {'Test Acc':<10}")
    print("-"*70)

    for model_name, result in results.items():
        model_display = model_name.replace('_', ' ').title()
        prediction = result['prediction'].upper()

        if result['confidence']:
            confidence = f"{result['confidence']:.1f}%"
        else:
            confidence = "N/A"

        test_acc = f"{result['test_accuracy']*100:.1f}%"

        print(f"{model_display:<25} {prediction:<10} {confidence:<15} {test_acc:<10}")

    # Consensus
    spam_votes = sum(1 for r in results.values() if r['prediction'] == 'spam')
    ham_votes = len(results) - spam_votes

    print("-"*70)
    print(f"📊 Consensus: {spam_votes} models say SPAM, {ham_votes} models say HAM")

    if spam_votes > ham_votes:
        print("🚨 FINAL VERDICT: SPAM")
    elif ham_votes > spam_votes:
        print("✅ FINAL VERDICT: HAM (Legitimate)")
    else:
        print("⚖️  FINAL VERDICT: TIE (Uncertain)")
    print("="*70)

#### 6. Test with sample messages
def test_predictions():
    # Test messages
    test_messages = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!",
        "Hey, can we meet for coffee tomorrow at 3pm?",
        "URGENT! Your account will be suspended. Verify now: http://fake-link.com",
        "Hi mom, I'll be home late tonight. Don't wait for dinner.",
        "Get rich quick! Make $5000 per week working from home!!!",
        "Meeting rescheduled to 2pm in conference room B"
    ]

    for message in test_messages:
        cleaned, results = predict_all_models(message)
        display_results(message, cleaned, results)
        print("\n")

#### 7. Interactive mode
def interactive_mode():
    print("\n🤖 Interactive Spam Detection Mode")
    print("Type your message or 'quit' to exit\n")

    while True:
        message = input("📧 Enter message: ").strip()

        if message.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not message:
            print("⚠️  Please enter a message\n")
            continue

        cleaned, results = predict_all_models(message)
        display_results(message, cleaned, results)
        print("\n")

#### 8. Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command-line mode: python predict_sms.py "Your message here"
        message = ' '.join(sys.argv[1:])
        cleaned, results = predict_all_models(message)
        display_results(message, cleaned, results)
    else:
        # Test mode first, then interactive
        print("🧪 Running test predictions...\n")
        test_predictions()

        # Ask if user wants interactive mode
        choice = input("\n💬 Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode()
