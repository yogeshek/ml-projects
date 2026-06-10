from pydantic import BaseModel

class Message(BaseModel):
    text: str
    
# GLOBAL VARIABLE FOR MODEL ###############################

model = None
Vectorizer = None

# ASYNC CONTEXT MANAGER- RUNS SETUP CODE BEFORE BLOCK AND CLEANUP AFTER THE BLOCK ####################################
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
@asynccontextmanager
async def lifespan(_):
    global model, Vectorizer
    try:
        model_path = Path(__file__).parent.parent.parent.parent / "models"
        #load vectorizer
        Vectorizer = joblib.load(model_path / "my_first_spam_vectorizer.pkl")
        #load model
        model = joblib.load(model_path / "my_first_spam_classifier.pkl")
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    
    #shutting down: cleanup if needed
    print("shutting down....")
    
# CREATE API END POINTS USING FAST API ######################
from fastapi import FastAPI, HTTPException
app = FastAPI(title="my first spam", version="1.1.1.0", lifespan=lifespan)

from my_first_train_spam import clean_text

@app.get("/health")
def health():
    return{
        "status" : "healthy" if model else "unhealthy",
        "model_loaded" : model is not None,
        "vectorizer_loaded": Vectorizer is not None
    }
@app.post("/predict")
def predict(msg: Message): #msg parameter gets assigned the Message(class) object.
    # exception handling
    if not model or not Vectorizer:
        raise HTTPException(
            status_code= 503,
            detail="model not loaded check the server log"
        )
    try:
        cleaned = clean_text(msg.text)
        text_vectorized = Vectorizer.transform([cleaned])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        return{
            "message": msg.text,
            "cleand": cleaned,
            "prediction":"spam" if prediction==1 else "ham",
            "confidence": f"{probabilities[prediction]*100:.2f}",
            "probabilities": {
                "ham":f"{probabilities[0] * 100:.2f}",
                "spam":f"{probabilities[1] * 100:.2f}"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail = f"prediction error: {str(e)}"
        )
# RUN THE SERVER ######################################3
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
    


