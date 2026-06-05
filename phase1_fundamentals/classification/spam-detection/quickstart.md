# 🚀 Quick Start Guide

Get your production ML API running in 5 minutes!

---

## **Step 1: Install Dependencies** (1 min)

```bash
pip install -r requirements.txt
```

---

## **Step 2: Train Models** (2 min)

```bash
python train_sms.py
```

**Output:**
```
✅ Saved vectorizer to models/vectorizer.pkl
✅ Saved naive_bayes to models/naive_bayes.pkl
   Test Accuracy: 98.3%
✅ Saved random_forest to models/random_forest.pkl
   Test Accuracy: 97.8%
...
🎉 All models saved successfully!
```

---

## **Step 3: Start API** (30 seconds)

```bash
python api.py
```

**You'll see:**
```
🚀 Starting SMS Spam Detection API...
✅ Loaded naive_bayes (Test Acc: 98.3%)
✅ Loaded random_forest (Test Acc: 97.8%)
✅ Loaded logistic_regression (Test Acc: 98.7%)
✅ Loaded svc (Test Acc: 98.5%)

🎉 Successfully loaded 4 models!
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## **Step 4: Test API** (1 min)

### Option A: Open Browser
Go to: http://localhost:8000/docs

### Option B: Run Test Script
```bash
# In a new terminal
python test_api.py
```

### Option C: Use cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! Free iPhone! Click now!"}'
```

**Response:**
```json
{
  "success": true,
  "message": "WINNER! Free iPhone! Click now!",
  "final_verdict": "SPAM",
  "consensus": {"spam": 4, "ham": 0},
  "predictions": {
    "naive_bayes": {
      "prediction": "spam",
      "confidence": 99.8,
      "model_accuracy": 98.3
    },
    ...
  },
  "processing_time_ms": 12.5
}
```

---

## **🎉 That's it! Your ML API is running!**

### **What you just built:**

✅ Production-ready REST API  
✅ 4 ML models working together  
✅ Automatic documentation (Swagger UI)  
✅ Request validation  
✅ Error handling  
✅ Health checks  

---

## **Next Steps:**

### **Learn Docker** (10 min)
```bash
docker-compose up -d
```
Access at: http://localhost:8000

### **Try RAG Chatbot** (5 min)
```bash
python rag_chatbot.py
```

### **Deploy to Cloud** (30 min)
See: `DEPLOYMENT_GUIDE.md`

---

## **Common Commands:**

```bash
# Start API
python api.py

# Start with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Test API
python test_api.py

# RAG demo
python rag_chatbot.py
```

---

## **Troubleshooting:**

**Port already in use?**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

**Models not found?**
```bash
# Train models first
python train_sms.py
```

**Import errors?**
```bash
# Install dependencies
pip install -r requirements.txt
```

---

## **API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/models` | GET | Model info |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | Swagger UI |

---

## **Example API Calls:**

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"message": "Win FREE money now!"}
)
print(response.json()['final_verdict'])  # SPAM
```

### JavaScript
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: 'Win FREE money!'})
})
.then(r => r.json())
.then(data => console.log(data.final_verdict));
```

### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi mom"}'
```

---

## **Performance:**

- **Response time**: ~10-15ms per prediction
- **Throughput**: ~100 requests/second (single instance)
- **Model size**: ~2MB (all 4 models + vectorizer)
- **Memory**: ~200MB RAM

---

## **Skills You Just Learned:**

✅ REST API development (FastAPI)  
✅ ML model deployment  
✅ Docker containerization  
✅ API documentation (OpenAPI/Swagger)  
✅ Request validation (Pydantic)  
✅ Ensemble ML (4 models voting)  

**This is production ML engineering!** 🚀

---

**Questions? Check:** `DEPLOYMENT_GUIDE.md` for detailed explanations
