"""
FastAPI REST API for SMS Spam Detection
Production-ready API with all 4 trained models
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pathlib import Path
import joblib
import re
from nltk.stem import PorterStemmer
from typing import Dict, List, Optional
import time
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="SMS Spam Detection API",
    description="Multi-model spam detection system using ML",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc UI
)

# Add CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'models'

# Global variables for models (loaded once at startup)
vectorizer = None
models = {}
stemmer = PorterStemmer()

# ============== REQUEST/RESPONSE MODELS ==============

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="SMS message to classify",
        example="Congratulations! You've won a FREE iPhone. Click here now!"
    )

    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()

class ModelPrediction(BaseModel):
    """Single model prediction result"""
    prediction: str = Field(description="ham or spam")
    confidence: Optional[float] = Field(None, description="Confidence score (0-100)")
    model_accuracy: float = Field(description="Model's test accuracy")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    message: str
    cleaned_message: str
    predictions: Dict[str, ModelPrediction]
    consensus: Dict[str, int]
    final_verdict: str
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: int
    vectorizer_loaded: bool
    timestamp: str

class ModelsInfoResponse(BaseModel):
    """Response model for models info endpoint"""
    total_models: int
    models: Dict[str, Dict[str, float]]

# ============== HELPER FUNCTIONS ==============

def clean_text(text: str) -> str:
    """Preprocess text (same as training)"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def load_models():
    """Load all models at startup"""
    global vectorizer, models

    try:
        # Load vectorizer
        vectorizer_path = MODELS_PATH / 'vectorizer.pkl'
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)

        # Load all models
        model_files = ['naive_bayes', 'random_forest', 'logistic_regression', 'svc']
        for model_name in model_files:
            model_path = MODELS_PATH / f'{model_name}.pkl'
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                print(f"[OK] Loaded {model_name}")
            else:
                print(f"[WARNING] Model not found: {model_name}")

        if len(models) == 0:
            raise RuntimeError("No models loaded. Please train models first.")

        print(f"\n[SUCCESS] Successfully loaded {len(models)} models!")
        return True

    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return False

def predict_message(message: str) -> Dict:
    """Run prediction across all models"""
    start_time = time.time()

    # Preprocess
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])

    # Get predictions from all models
    results = {}
    spam_votes = 0
    ham_votes = 0

    for model_name, model_data in models.items():
        model = model_data['model']

        # Prediction
        prediction = model.predict(vectorized)[0]
        prediction_label = 'spam' if prediction == 1 else 'ham'

        # Count votes
        if prediction_label == 'spam':
            spam_votes += 1
        else:
            ham_votes += 1

        # Probability (if available)
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized)[0][1]
            confidence = proba * 100

        results[model_name] = ModelPrediction(
            prediction=prediction_label,
            confidence=confidence,
            model_accuracy=model_data['test_accuracy'] * 100
        )

    # Final verdict
    if spam_votes > ham_votes:
        verdict = "SPAM"
    elif ham_votes > spam_votes:
        verdict = "HAM"
    else:
        verdict = "UNCERTAIN"

    processing_time = (time.time() - start_time) * 1000  # Convert to ms

    return {
        'cleaned_message': cleaned,
        'predictions': results,
        'consensus': {'spam': spam_votes, 'ham': ham_votes},
        'final_verdict': verdict,
        'processing_time_ms': round(processing_time, 2)
    }

# ============== STARTUP/SHUTDOWN EVENTS ==============

@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    print("[STARTUP] Starting SMS Spam Detection API...")
    success = load_models()
    if not success:
        print("[WARNING] API started but models not loaded. Please check model files.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("[SHUTDOWN] Shutting down SMS Spam Detection API...")

# ============== API ENDPOINTS ==============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "SMS Spam Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if models else "unhealthy",
        models_loaded=len(models),
        vectorizer_loaded=vectorizer is not None,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/models", response_model=ModelsInfoResponse, tags=["Models"])
async def get_models_info():
    """Get information about loaded models"""
    models_info = {}
    for model_name, model_data in models.items():
        models_info[model_name] = {
            'test_accuracy': round(model_data['test_accuracy'] * 100, 2),
            'test_precision': round(model_data['test_precision'] * 100, 2),
            'test_recall': round(model_data['test_recall'] * 100, 2),
            'cv_accuracy': round(model_data['cv_accuracy'] * 100, 2)
        }

    return ModelsInfoResponse(
        total_models=len(models),
        models=models_info
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict if a message is spam or ham

    - **message**: SMS message to classify (1-500 characters)

    Returns predictions from all models with consensus verdict
    """
    # Check if models are loaded
    if not models or not vectorizer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please ensure models are trained and available."
        )

    try:
        # Run prediction
        result = predict_message(request.message)

        return PredictionResponse(
            success=True,
            message=request.message,
            cleaned_message=result['cleaned_message'],
            predictions=result['predictions'],
            consensus=result['consensus'],
            final_verdict=result['final_verdict'],
            processing_time_ms=result['processing_time_ms'],
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(messages: List[str]):
    """
    Batch prediction for multiple messages

    - **messages**: List of SMS messages to classify

    Returns predictions for all messages
    """
    if not models or not vectorizer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded."
        )

    if len(messages) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 messages per batch request"
        )

    results = []
    for message in messages:
        if message.strip():
            result = predict_message(message)
            results.append({
                'message': message,
                'verdict': result['final_verdict'],
                'consensus': result['consensus']
            })

    return {
        'success': True,
        'total_messages': len(messages),
        'results': results,
        'timestamp': datetime.utcnow().isoformat()
    }

# ============== RUN SERVER ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (dev only)
    )
