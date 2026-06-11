from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import sys
from pathlib import Path

# Add the spam-detection directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# pydantic BaseModel - its a foundation class for creating and validation and serialization models in python.
class Message(BaseModel):
    text: str
    @field_validator('text') # @ is decorator register the funciton with pydantic, and pydantic calls the function with the two parameters
    def validate_text(cls, v):
        if len(v.strip())==0:
            raise ValueError(f"Text cannot be empty")
        if len(v) > 10000:
            raise ValueError('Text is too long')
        return v

###################################################################

models = {}
vectorizer = None
from contextlib import asynccontextmanager
from pathlib import Path
import joblib

@asynccontextmanager
async def lifespan(_):
    global vectorizer, models
    try:
        model_path = Path(__file__).resolve().parent.parent.parent.parent / "models"
        print(f"Loading models from: {model_path}")

        vectorizer = joblib.load(model_path/"my_first_spam_ensemble_vectorizer.pkl")
        print(f"[OK] Vectorizer loaded")

        model_name = ['my_naive_bayes', 'my_random_forest',
                      'my_logistic_regression', 'my_svc']

        for name in model_name:
            model_data = joblib.load(model_path/f"{name}.pkl")
            models[name]= model_data
            print(f"[OK] Loaded {name}")

        print(f"\n[SUCCESS] Models loaded: {list(models.keys())}")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise
    yield
    print("Shutting down...")

app = FastAPI(title="my ensemble api", version="1.1.1.0", lifespan=lifespan)

# Define clean_text function directly to avoid import issues
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = ' '.join(text.split())
    return text

@app.post("/predict_ensemble")
def predict_ensemble(msg: Message):
    if not models or not vectorizer:
        raise HTTPException(
            status_code=503, detail="Models not loaded"
        )

    try:
        cleaned = clean_text(msg.text)
        vectorized = vectorizer.transform([cleaned])

        predictions = {}
        votes = {"spam":0, "ham":0}

        for name, model_data in models.items():
            model = model_data['model']
            pred = model.predict(vectorized)[0]
            pred_label = "spam" if pred == 1 else "ham"

            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(vectorized)[0][1]*100
            else:
                prob = None
            predictions[name] ={
                "prediction": pred_label,
                "confidence": prob,
                "model_accuracy": model_data["test_accuracy"]*100
            }

            votes[pred_label] +=1
        final_verdict = "spam" if votes["spam"]>votes["ham"] else "ham"

        return {
            "message": msg.text,
            "cleaned" : cleaned,
            "predictions" : predictions,
            "votes": votes,
            "final_verdict": final_verdict

        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Error in prediction: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
    
#run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006) 
                   
        


        
            
