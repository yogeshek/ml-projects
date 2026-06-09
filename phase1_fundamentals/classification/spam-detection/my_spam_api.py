from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from typing import List
import re

# Pydantic model for request
class Message(BaseModel):
    text: str
    
################################# **Goal:** one models for predicting

# Global variable for model
# model = None
# vectorizer = None

# @asynccontextmanager
# async def lifespan(_):
#     # Startup: Load models
#     global model, vectorizer

#     try:
#         #update the path where the models are
#         models_path = Path(__file__).parent.parent.parent.parent / "models"

#         print(f"Loading models from : {models_path}")

#         #load vectorizer
#         vectorizer = joblib.load(models_path / "spam_vectorizer.pkl")

#         #load models
#         model_data = joblib.load(models_path / "spam_classifier.pkl")
        
#         # if model save directly in the train.py
        
#         # model = model_data

#         # Check if it's a dictionary or the model itself
#         if isinstance(model_data, dict):
#             model = model_data['model']
#             accuracy = model_data.get('test_accuracy', 0)
#             print(f" model loaded (Accuracy: {accuracy*100:.1f}%)")
#         else:
#             model = model_data
#             print(" model loaded")
#     except Exception as e:
#         print(f" Error loading models: {e}")
#         raise

#     yield

#     # Shutdown: cleanup if needed
#     print("Shutting down...")

# app = FastAPI(title="my spam api", version="1.1.1.0", lifespan=lifespan)

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]','',text)
#     text = ' '.join(text.split())
#     return text

# @app.get("/health")
# def health():
#     return{
#         "status" : "healthy" if model else  "unhealthy",
#         "model_loaded" : model is not None,
#         "vectorizer_loaded" : vectorizer is not None
#     }
    
# @app.post("/predict")
# def predict(msg: Message):
#     if not model or not vectorizer:
#         raise HTTPException(
#             status_code = 503,
#             detail = "Model not loaded check the server log"
#         )
#     try:
#         cleaned = clean_text(msg.text)
#         text_vectorized = vectorizer.transform([cleaned])
#         prediction = model.predict(text_vectorized)[0]
#         probabilities = model.predict_proba(text_vectorized)[0]

#         return{
#             "message": msg.text,
#             "cleaned": cleaned,
#             "prediction": "spam" if prediction == 1 else "ham",
#             "confidence": float(probabilities[prediction]*100),
#             "probabilities": {
#                 "ham": float(probabilities[0]*100),
#                 "spam": float(probabilities[1]*100)
#             }

#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"prediction error: {str(e)}"
#         )
    
# #run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8002)
    
    
################################# **Goal:** Use all your models with ensemble voting

models = {}
vectorizer = None

@asynccontextmanager
async def lifespan(_):
    global vectorizer
    try:
        model_path = Path(__file__).parent.parent.parent.parent / "models"
        print(f"loading models from : {model_path}")

        vectorizer = joblib.load(model_path/"vectorizer.pkl")
        print(f"vectorizer loaded")
        
        model_names = ['naive_bayes', 'random_forest', 
                      'logistic_regression', 'svc']
        
        for name in model_names:
            model_data = joblib.load(model_path/f"{name}.pkl")
            models[name]= model_data
            # acc = model_data['test_accuracy']
            # print(f"{name}: {acc * 100:.1f} accuracy")
        print(f" loaded {len(models)} models successfully")
    
    except Exception as e:
        print(f"Error: {e}")
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
        "status" : "healthy" if models else  "unhealthy",
        "model_loaded" : models is not None,
        "vectorizer_loaded" : vectorizer is not None
    }    
        
#FOR single message
        
@app.post("/predict_ensemble")
def predict_ensemble(msg: Message):
    if not models or not vectorizer:
        raise HTTPException(status_code=503, detail="no model found")
    
    try:
        cleaned = clean_text(msg.text)
        vectorized = vectorizer.transform([cleaned])

        predictions = {}
        votes ={"spam":0, "ham":0}
        
        for name, model_data in models.items():
            model = model_data['model']
            #predict
            prediction_all = model.predict(vectorized)
            pred = model.predict(vectorized)[0]
            pred_label = "spam" if pred==1 else "ham"
            #get probability if available
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(vectorized)[0][1]*100
            else:
                prob = None
                
            predictions[name]= {
                "predictions": pred_label,
                "confidence": prob,
                "model_accuracy": model_data["test_accuracy"] * 100
            }
            
            #count votes
            votes[pred_label] += 1
        #determine final predict
        final_verdict = "spam" if votes["spam"]>votes["ham"] else "ham"
        
        return {
            "message": msg.text,
            "cleaned" : cleaned,
            "predictions" : predictions,
            "votes": votes,
            "final_verdict": final_verdict
            
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# for multiple message
@app.post("/predict_ensemble_batch")
def predict_ensemble_batch(messages: List[Message]):
    if not models or not vectorizer:
        raise HTTPException(status_code=503, detail="no model found")

    try:
        cleaned_message = [clean_text(m.text) for m in messages]
        vectorized_all = vectorizer.transform(cleaned_message)
        results = []

        for msg_idx in range(len(messages)):
            predictions = {}
            votes ={"spam":0, "ham":0}
            
            for name, model_data in models.items():
                model = model_data['model']
                
                pred = model.predict(vectorized_all)[msg_idx]
                pred_label = "spam" if pred ==1 else "ham"
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(vectorized_all)[msg_idx][1]*100
                else:
                    prob = None
                predictions[name] = {
                    "predictions": pred_label,
                    "confidence": prob,
                    "model_accuracy": model_data["test_accuracy"]*100
                }
                
                votes[pred_label] += 1
            final_verdict = "spam" if votes["spam"] > votes["ham"] else "ham"
            
            results.append({
                "message" : messages[msg_idx].text,
                "cleaned" : cleaned_message[msg_idx],
                "predictions": predictions,
                "votes": votes,
                "final_verdict": final_verdict
            })
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
                                
    
#run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)        
        


