#1. Import
# # pickle to load the saved model and vectorizer files
# re:  for text cleaning
import pickle
import re
import sys
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join(text.split())
    return text

#########
def load_models():
    """load the trained model and vectorizer"""
    try:
        model_file = MODELS_PATH / 'spam_classifier.pkl'
        vectorizer_file = MODELS_PATH / 'vectorizer.pkl'

        with open(model_file,'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_file,'rb') as f:
            vectorizer = pickle.load(f)

        print("model loaded successfully!\n")
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found!")
        print(f"Expected files at: {MODELS_PATH}")
        print("please run train.py to load the models")
        sys.exit(1)    
        
########
def predict_single_message(message, model, vectorizer, show_details=True):
    """predict if a single message is spam or ham"""
    
    cleaned_message = clean_text(message)
    message_vec = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0]
    
    ham_confidence = probability[0]*100
    spam_confidence = probability[1]*100
    
    if show_details:
        print(f"\nPrediction: {'SPAM' if prediction ==1 else 'HAM'}")
        print(f" Ham: {ham_confidence:.2f}%")
        print(f" Spam: {spam_confidence:.2f}%")
    return prediction, spam_confidence

########
def predict_batch(messages, model, vectorizer):
    """predict multiple messages in batch"""
    results=[]
    for i, msg in enumerate(messages, 1):
        cleaned_message = clean_text(msg)
        message_vec = vectorizer.transform([cleaned_message])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]

        result = {
            'message': msg,
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'spam_confidence': probability[1]*100
        }
        results.append(result)

        status_emoji = "🚨" if prediction == 1 else "✅"
        print(f"{i}. {status_emoji} {result['prediction']} ({result['spam_confidence']:.1f}%)- {msg[:50]}.......")

    return results

#######

def interactive_model(model, vectorizer):
    while True:
        try:
            message = input("Enter your message interactive: ").strip()
            
            if message.lower() in ['quit','exit','q']:
                print("\nGood Bye")
                break
            if not message:
                continue
            predict_single_message(message, model, vectorizer)
        except KeyboardInterrupt:
            print("\n\nGoodbye")
            break
        
def main():
    model, vectorizer = load_models()
    #show menu
    print("==== Spam Classifier ====")
    print("1. Single prediction")
    print("2. Batch prediction")
    print("3. Interactive mode")

    choice = input("Enter your choice: ")

    if choice == '1':
        message = input("Enter your message: ")
        predict_single_message(message, model, vectorizer)
    elif choice == '2':
        message = input("Enter messages (comma-separated): ")
        message_list = [msg.strip() for msg in message.split(",") if msg.strip()]
        predict_batch(message_list, model, vectorizer)
    elif choice == '3':
        interactive_model(model, vectorizer)
    else:
        print("No valid input provided")
        
if __name__ == "__main__":
    main()
   