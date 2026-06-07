from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
import re

# Pydantic model for request
class Message(BaseModel):
    text: str

# Global variable for model
model = None
vectorizer = None

@asynccontextmanager
async def lifespan(_):
    # Startup: Load models
    global model, vectorizer

    try:
        #update the path where the models are
        models_path = Path(__file__).parent.parent.parent.parent / "models"

        print(f"Loading models from : {models_path}")

        #load vectorizer
        vectorizer = joblib.load(models_path / "spam_vectorizer.pkl")

        #load models
        model_data = joblib.load(models_path / "spam_classifier.pkl")
        
        # if model save directly in the train.py
        
        # model = model_data

        # Check if it's a dictionary or the model itself
        if isinstance(model_data, dict):
            model = model_data['model']
            accuracy = model_data.get('test_accuracy', 0)
            print(f" model loaded (Accuracy: {accuracy*100:.1f}%)")
        else:
            model = model_data
            print(" model loaded")
    except Exception as e:
        print(f" Error loading models: {e}")
        raise

    yield

    # Shutdown: cleanup if needed
    print("Shutting down...")

app = FastAPI(title="my spam api", version="1.1.1.0", lifespan=lifespan)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join(text.split())
    return text

@app.get("/health")
def health():
    return{
        "status" : "healthy" if model else  "unhealthy",
        "model_loaded" : model is not None,
        "vectorizer_loaded" : vectorizer is not None
    }
    
@app.post("/predict")
def predict(msg: Message):
    if not model or not vectorizer:
        raise HTTPException(
            status_code = 503,
            detail = "Model not loaded check the server log"
        )
    try:
        cleaned = clean_text(msg.text)
        text_vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        return{
            "message": msg.text,
            "cleaned": cleaned,
            "prediction": "spam" if prediction == 1 else "ham",
            "confidence": float(probabilities[prediction]*100),
            "probabilities": {
                "ham": float(probabilities[0]*100),
                "spam": float(probabilities[1]*100)
            }

        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction error: {str(e)}"
        )
    
#run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)

