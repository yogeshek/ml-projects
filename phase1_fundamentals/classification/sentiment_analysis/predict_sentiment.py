import pickle
import re
import sys
from pathlib import Path

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

def clean_text(text):
    """Clean text using same preprocessing as training"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join(text.split())
    return text

def load_models():
    """Load the trained sentiment model and vectorizer"""
    try:
        model_file = MODELS_PATH / 'sentiment_classifier.pkl'
        vectorizer_file = MODELS_PATH / 'sentiment_vectorizer.pkl'

        with open(model_file,'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_file,'rb') as f:
            vectorizer = pickle.load(f)

        print("Model loaded successfully!\n")
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found!")
        print(f"Expected files at: {MODELS_PATH}")
        print("Please run train_sentiment.py first to create the models")
        sys.exit(1)

def predict_single_review(review, model, vectorizer, show_details=True):
    """Predict sentiment of a single review"""
    cleaned_review = clean_text(review)
    review_vec = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vec)[0]
    probability = model.predict_proba(review_vec)[0]

    positive_confidence = probability[0]*100
    negative_confidence = probability[1]*100

    if show_details:
        print(f"\nPrediction: {'NEGATIVE' if prediction == 1 else 'POSITIVE'}")
        print(f" Positive: {positive_confidence:.2f}%")
        print(f" Negative: {negative_confidence:.2f}%")

    return prediction, negative_confidence

def predict_batch(reviews, model, vectorizer):
    """Predict sentiment for multiple reviews in batch"""
    results = []
    for i, review in enumerate(reviews, 1):
        cleaned_review = clean_text(review)
        review_vec = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vec)[0]
        probability = model.predict_proba(review_vec)[0]

        result = {
            'review': review,
            'prediction': 'NEGATIVE' if prediction == 1 else 'POSITIVE',
            'negative_confidence': probability[1]*100
        }
        results.append(result)

        status_emoji = "😞" if prediction == 1 else "😊"
        print(f"{i}. {status_emoji} {result['prediction']} ({result['negative_confidence']:.1f}%) - {review[:50]}...")

    return results

def interactive_mode(model, vectorizer):
    """Interactive mode for testing reviews"""
    print("\n=== Interactive Sentiment Analysis ===")
    print("Type 'quit', 'exit', or 'q' to exit\n")

    while True:
        try:
            review = input("Enter your review: ").strip()

            if review.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not review:
                continue

            predict_single_review(review, model, vectorizer)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

def main():
    model, vectorizer = load_models()

    print("==== Sentiment Classifier ====")
    print("1. Single prediction")
    print("2. Batch prediction")
    print("3. Interactive mode")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        review = input("Enter your review: ")
        predict_single_review(review, model, vectorizer)

    elif choice == '2':
        reviews_input = input("Enter reviews (comma-separated): ")
        review_list = [r.strip() for r in reviews_input.split(",") if r.strip()]
        predict_batch(review_list, model, vectorizer)

    elif choice == '3':
        interactive_mode(model, vectorizer)

    else:
        print("Invalid choice! Please select 1, 2, or 3")

if __name__ == "__main__":
    main()
