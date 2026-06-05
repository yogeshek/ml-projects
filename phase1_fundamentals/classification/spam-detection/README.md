# 🚀 Production ML System: SMS Spam Detection

**From ML Model to Production API in 3 Steps**

A complete production-ready machine learning system featuring:
- 🎯 4 ensemble ML models (98%+ accuracy)
- 🔌 FastAPI REST API
- 🐳 Docker deployment
- 🤖 RAG chatbot (Generative AI demo)
- ☁️ Cloud-ready architecture

---

## 📁 **Project Structure**

```
spam-detection/
├── 📊 Training & Prediction
│   ├── train_sms.py          # Train 4 models, save as .pkl files
│   ├── predict_sms.py        # Local prediction script
│
├── 🔌 Production API
│   ├── api.py                # FastAPI REST API (production-ready)
│   ├── test_api.py           # API test suite
│   ├── requirements.txt      # Python dependencies
│
├── 🐳 Docker Deployment
│   ├── Dockerfile            # Multi-stage Docker build
│   ├── docker-compose.yml    # API + Redis + Nginx
│   ├── nginx.conf            # Reverse proxy config
│   └── .dockerignore         # Docker ignore rules
│
├── 🤖 Generative AI
│   └── rag_chatbot.py        # RAG pattern demo with Claude API
│
└── 📚 Documentation
    ├── README.md             # This file
    ├── quickstart.md         # Get started in 5 minutes
    ├── DEPLOYMENT_GUIDE.md   # Complete deployment guide
    └── SKILLS_SUMMARY.md     # Skills assessment
```

---

## ⚡ **Quick Start** (5 minutes)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train Models
```bash
python train_sms.py
```

### 3️⃣ Start API
```bash
python api.py
```

### 4️⃣ Test API
Open browser: http://localhost:8000/docs

Or use cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! Free iPhone!"}'
```

**📖 Detailed guide:** See [quickstart.md](quickstart.md)

---

## 🎯 **Features**

### **Machine Learning**
- ✅ 4 trained models (Naive Bayes, Random Forest, LR, SVC)
- ✅ Ensemble voting with consensus
- ✅ 98%+ accuracy on spam detection
- ✅ TF-IDF + bigrams feature extraction
- ✅ Cross-validation for robust evaluation
- ✅ Model persistence with performance metrics

### **Production API**
- ✅ FastAPI REST endpoints
- ✅ Automatic API documentation (Swagger/OpenAPI)
- ✅ Request validation (Pydantic)
- ✅ Error handling & health checks
- ✅ Batch processing support
- ✅ CORS enabled
- ✅ Response time tracking

### **Deployment**
- ✅ Docker containerization
- ✅ Docker Compose orchestration
- ✅ Nginx reverse proxy
- ✅ Redis caching support
- ✅ Health check monitoring
- ✅ Multi-cloud ready (AWS, Azure, GCP)

### **Generative AI**
- ✅ RAG (Retrieval Augmented Generation) pattern
- ✅ Vector search with semantic similarity
- ✅ Claude API integration
- ✅ Knowledge base + retrieval + generation

---

## 📊 **Model Performance**

| Model | Test Accuracy | CV Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-------------|-----------|--------|----------|
| Naive Bayes | 98.3% | 97.2% | 97.5% | 96.8% | 97.1% |
| Random Forest | 97.8% | 97.0% | 96.2% | 96.5% | 96.3% |
| Logistic Regression | **98.7%** | **98.1%** | 98.2% | 97.9% | 98.0% |
| SVC | 98.5% | 97.8% | 97.8% | 97.5% | 97.6% |

**API Performance:**
- Response time: 10-15ms per prediction
- Throughput: ~100 requests/second
- Memory usage: ~200MB RAM

---

## 🔌 **API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/models` | Model metadata & performance |
| POST | `/predict` | Single message prediction |
| POST | `/predict/batch` | Batch predictions (max 100) |
| GET | `/docs` | Interactive API docs (Swagger) |
| GET | `/redoc` | Alternative API docs (ReDoc) |

### **Example Request:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"message": "Congratulations! You won $1000!"}
)

print(response.json())
```

### **Example Response:**

```json
{
  "success": true,
  "message": "Congratulations! You won $1000!",
  "cleaned_message": "congratul you won",
  "predictions": {
    "naive_bayes": {
      "prediction": "spam",
      "confidence": 99.8,
      "model_accuracy": 98.3
    },
    "random_forest": { "prediction": "spam", "confidence": 95.2, "model_accuracy": 97.8 },
    "logistic_regression": { "prediction": "spam", "confidence": 98.1, "model_accuracy": 98.7 },
    "svc": { "prediction": "spam", "confidence": null, "model_accuracy": 98.5 }
  },
  "consensus": {"spam": 4, "ham": 0},
  "final_verdict": "SPAM",
  "processing_time_ms": 12.5,
  "timestamp": "2025-06-05T10:30:45.123Z"
}
```

---

## 🐳 **Docker Deployment**

### **Single Container:**
```bash
docker build -t spam-api:v1.0 .
docker run -d -p 8000:8000 --name spam-api spam-api:v1.0
```

### **Full Stack (Recommended):**
```bash
docker-compose up -d
```

This starts:
- **spam-api** (port 8000) - FastAPI application
- **redis** (port 6379) - Caching layer
- **nginx** (port 80) - Reverse proxy

### **Check Status:**
```bash
docker-compose ps
docker-compose logs -f spam-api
```

---

## 🤖 **RAG Chatbot (Generative AI)**

Learn modern AI patterns with our RAG implementation:

```bash
# Vector search demo (no API key needed)
python rag_chatbot.py

# Full RAG with Claude API
export ANTHROPIC_API_KEY='your-key'
python rag_chatbot.py --full

# Interactive mode
python rag_chatbot.py --interactive
```

**What you'll learn:**
- Vector embeddings & semantic search
- Retrieval Augmented Generation (RAG)
- LLM integration (Claude API)
- Prompt engineering with context

---

## ☁️ **Cloud Deployment**

Deploy to any cloud platform:

### **AWS:**
- **EC2**: Docker container on virtual machine
- **ECS**: Managed container orchestration
- **Lambda**: Serverless deployment

### **Azure:**
- **App Service**: Container deployment
- **Container Instances**: Quick container hosting
- **AKS**: Kubernetes orchestration

### **Google Cloud:**
- **Cloud Run**: Serverless containers
- **GKE**: Kubernetes clusters
- **Compute Engine**: VM instances

**📖 Complete guide:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 📚 **Documentation**

- **[quickstart.md](quickstart.md)** - Get started in 5 minutes
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment (API + Docker + Cloud + RAG)
- **[SKILLS_SUMMARY.md](SKILLS_SUMMARY.md)** - Skills assessment & career roadmap
- **API Docs**: http://localhost:8000/docs (when running)

---

## 🛠️ **Technology Stack**

**Machine Learning:**
- scikit-learn (classical ML algorithms)
- NLTK (text preprocessing)
- Pandas (data manipulation)
- NumPy (numerical operations)

**API & Backend:**
- FastAPI (modern Python web framework)
- Pydantic (data validation)
- Uvicorn (ASGI server)

**Deployment:**
- Docker (containerization)
- Docker Compose (orchestration)
- Nginx (reverse proxy)
- Redis (caching)

**Generative AI:**
- Anthropic Claude API (LLM)
- TF-IDF vectorization (embeddings)
- Cosine similarity (semantic search)

---

## 📈 **Project Progression**

```
Phase 1: ML Fundamentals ✅
├── Data preprocessing
├── Feature engineering (TF-IDF)
├── Train 4 models
├── Cross-validation
└── Model evaluation

Phase 2: Production API ✅
├── FastAPI REST endpoints
├── Request validation
├── API documentation
├── Error handling
└── Testing suite

Phase 3: Containerization ✅
├── Dockerfile (multi-stage build)
├── Docker Compose
├── Nginx reverse proxy
├── Redis caching
└── Health checks

Phase 4: Generative AI ✅
├── RAG pattern
├── Vector search
├── LLM integration
└── Prompt engineering

Phase 5: Cloud Deploy 🔄
├── AWS/Azure/GCP guides
├── CI/CD pipeline (planned)
├── Monitoring & logging (planned)
└── Auto-scaling (planned)
```

---

## 🎓 **Skills Demonstrated**

This project showcases production ML engineering skills:

✅ **ML Engineering:**
- Model training & evaluation
- Feature engineering
- Ensemble methods
- Model persistence

✅ **Software Engineering:**
- REST API design
- Input validation
- Error handling
- Testing

✅ **DevOps:**
- Containerization (Docker)
- Orchestration (Docker Compose)
- Reverse proxy (Nginx)
- Health monitoring

✅ **AI/ML Deployment:**
- Model serving
- Inference optimization
- Batch processing
- API versioning

✅ **Modern AI:**
- RAG pattern
- Vector search
- LLM integration
- Prompt engineering

**📖 Detailed assessment:** See [SKILLS_SUMMARY.md](SKILLS_SUMMARY.md)

---

## 🚀 **Next Steps**

### **Immediate (Week 1-2):**
- [ ] Deploy to AWS EC2/ECS
- [ ] Add authentication (JWT)
- [ ] Implement rate limiting
- [ ] Set up monitoring (Prometheus/Grafana)

### **Short-term (Month 1-2):**
- [ ] Replace TF-IDF with BERT embeddings
- [ ] Fine-tune transformer model
- [ ] Add experiment tracking (MLflow)
- [ ] CI/CD pipeline (GitHub Actions)

### **Long-term (Month 3+):**
- [ ] Kubernetes deployment
- [ ] A/B testing framework
- [ ] Model drift monitoring
- [ ] Auto-retraining pipeline

---

## 💡 **Learning Resources**

**Documentation:**
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [Anthropic Claude](https://docs.anthropic.com/)

**Tutorials:**
- [ML Deployment Best Practices](https://madewithml.com/)
- [RAG Systems](https://www.pinecone.io/learn/rag/)
- [MLOps Guide](https://ml-ops.org/)

---

## 📞 **Support**

**Issues?**
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) troubleshooting section
- Review API logs: `docker-compose logs -f`
- Test endpoints: http://localhost:8000/docs

---

## 📄 **License**

MIT License - Feel free to use this project for learning and portfolio purposes.

---

## 🎉 **Acknowledgments**

- **Dataset**: SMS Spam Collection (UCI ML Repository)
- **Frameworks**: FastAPI, scikit-learn, Anthropic Claude
- **Community**: MLOps community, FastAPI Discord

---

**⭐ If this helped your ML journey, star the repo!**

**Built with ❤️ for ML Engineers transitioning to production**
