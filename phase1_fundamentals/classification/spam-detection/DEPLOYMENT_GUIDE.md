# 🚀 Production Deployment Guide

Complete guide to deploy your ML model as a production-ready API

---

## 📋 **Table of Contents**

1. [Step 1: FastAPI REST API](#step-1-fastapi-rest-api)
2. [Step 2: Docker Deployment](#step-2-docker-deployment)
3. [Step 3: RAG Chatbot (Generative AI)](#step-3-rag-chatbot)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Logging](#monitoring--logging)

---

## **STEP 1: FastAPI REST API** 🎯

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Train Models First

```bash
# Train and save all models
python train_sms.py

# This creates:
# - models/vectorizer.pkl
# - models/naive_bayes.pkl
# - models/random_forest.pkl
# - models/logistic_regression.pkl
# - models/svc.pkl
```

### Run the API

```bash
# Start the API server
python api.py

# Or with uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Test the API

```bash
# Run test script
python test_api.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/models` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

### Example Usage (cURL)

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! You won $1000! Call now!"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '["Free money!", "Hi mom", "URGENT! Click here"]'
```

### Example Usage (Python)

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"message": "WINNER! Free iPhone!"}
)
print(response.json())

# Response:
# {
#   "success": true,
#   "message": "WINNER! Free iPhone!",
#   "final_verdict": "SPAM",
#   "consensus": {"spam": 4, "ham": 0},
#   "predictions": {...},
#   "processing_time_ms": 12.5
# }
```

---

## **STEP 2: Docker Deployment** 🐳

### Why Docker?

- ✅ Consistent environment across dev/staging/prod
- ✅ Easy deployment to any cloud platform
- ✅ Isolated dependencies
- ✅ Scalable with orchestration (Kubernetes)

### Build Docker Image

```bash
# Make sure models are trained first!
python train_sms.py

# Build image
docker build -t spam-detection-api:v1.0 .

# Check image
docker images | grep spam-detection
```

### Run Container

```bash
# Run single container
docker run -d \
  --name spam-api \
  -p 8000:8000 \
  -v $(pwd)/../../../../models:/app/models:ro \
  spam-detection-api:v1.0

# Check logs
docker logs spam-api

# Stop container
docker stop spam-api
docker rm spam-api
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# This starts:
# - spam-api (port 8000)
# - redis (port 6379) - for caching
# - nginx (port 80) - reverse proxy

# Check status
docker-compose ps

# View logs
docker-compose logs -f spam-api

# Stop all services
docker-compose down
```

### Test Dockerized API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "Win FREE iPhone now!"}'
```

### Push to Docker Hub (Optional)

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag spam-detection-api:v1.0 yourusername/spam-detection-api:v1.0

# Push to registry
docker push yourusername/spam-detection-api:v1.0
```

---

## **STEP 3: RAG Chatbot (Generative AI)** 🤖

### What is RAG?

**RAG (Retrieval Augmented Generation)** combines:
1. **Retrieval**: Find relevant information from knowledge base
2. **Augmentation**: Add context to the prompt
3. **Generation**: LLM generates response based on context

### Setup

```bash
# Install dependencies
pip install anthropic  # For Claude API

# Set API key
export ANTHROPIC_API_KEY='your-api-key-here'

# Get your key from: https://console.anthropic.com/
```

### Run Demos

```bash
# 1. Vector search demo (no API key needed)
python rag_chatbot.py

# 2. Full RAG with Claude API
python rag_chatbot.py --full

# 3. Interactive chat mode
python rag_chatbot.py --interactive
```

### How It Works

```
User Query: "What is Naive Bayes?"
    ↓
┌─────────────────────────────────────┐
│  1. RETRIEVAL (Vector Search)       │
│  - Convert query to vector          │
│  - Search knowledge base            │
│  - Find top-k similar documents     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. AUGMENTATION                     │
│  - Add retrieved docs as context    │
│  - Build enhanced prompt            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. GENERATION (Claude/GPT)         │
│  - LLM generates response           │
│  - Grounded in retrieved context    │
└─────────────────────────────────────┘
    ↓
Response: "Naive Bayes is a probabilistic
classifier based on Bayes theorem..."
```

### Key Concepts You're Learning

1. **Vector Embeddings**: Converting text to numerical vectors
2. **Semantic Search**: Finding similar meaning (not just keywords)
3. **Cosine Similarity**: Measuring vector similarity
4. **LLM Integration**: Using Claude/GPT APIs
5. **Prompt Engineering**: Crafting effective prompts with context

### Production Vector Databases

Replace `SimpleVectorStore` with:
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source vector DB
- **ChromaDB**: Lightweight embeddings DB
- **FAISS**: Facebook's similarity search library

### Advanced RAG Patterns

```python
# 1. Hybrid Search (keyword + semantic)
# 2. Re-ranking results
# 3. Query expansion
# 4. Multi-hop reasoning
# 5. Conversational memory
```

---

## **Cloud Deployment** ☁️

### Option 1: AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
sudo apt update
sudo apt install docker.io docker-compose -y

# 4. Clone your code
git clone your-repo
cd spam-detection

# 5. Run with Docker Compose
docker-compose up -d

# 6. Configure security group (allow port 8000)
```

### Option 2: AWS ECS (Elastic Container Service)

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-ecr-url
docker tag spam-detection-api:v1.0 your-ecr-url/spam-api:v1.0
docker push your-ecr-url/spam-api:v1.0

# 2. Create ECS task definition
# 3. Create ECS service
# 4. Configure load balancer
```

### Option 3: AWS Lambda (Serverless)

```bash
# Use AWS Lambda + API Gateway for serverless deployment
# Good for: Low traffic, cost optimization
# Limitation: Cold starts, timeout limits
```

### Option 4: Azure App Service

```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name spam-api-rg --location eastus

# 3. Create App Service plan
az appservice plan create --name spam-api-plan --resource-group spam-api-rg --is-linux

# 4. Deploy container
az webapp create --resource-group spam-api-rg --plan spam-api-plan --name spam-api --deployment-container-image yourusername/spam-detection-api:v1.0
```

### Option 5: Google Cloud Run

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/spam-api

# 2. Deploy to Cloud Run
gcloud run deploy spam-api \
  --image gcr.io/PROJECT_ID/spam-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## **Monitoring & Logging** 📊

### Add Logging to API

```python
# In api.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Prediction request: {request.message[:50]}...")
    # ... prediction logic
    logger.info(f"Result: {result['final_verdict']}")
```

### Prometheus Metrics

```python
# Install: pip install prometheus-fastapi-instrumentator

from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
# Metrics available at /metrics
```

### Application Monitoring

- **Sentry**: Error tracking
- **DataDog**: APM & infrastructure monitoring
- **New Relic**: Application performance monitoring
- **Grafana + Prometheus**: Custom dashboards

---

## **Next Steps** 🎯

### Week 1-2: Production Hardening
- [ ] Add authentication (JWT tokens)
- [ ] Rate limiting (slowapi)
- [ ] Request validation
- [ ] Error handling
- [ ] Logging & monitoring

### Week 3-4: MLOps
- [ ] Experiment tracking (MLflow)
- [ ] Model versioning
- [ ] A/B testing
- [ ] Automated retraining
- [ ] CI/CD pipeline (GitHub Actions)

### Week 5-6: Scaling
- [ ] Load balancing
- [ ] Caching layer (Redis)
- [ ] Async processing (Celery)
- [ ] Kubernetes deployment
- [ ] Auto-scaling

### Week 7-8: Advanced AI
- [ ] Replace TF-IDF with BERT embeddings
- [ ] Fine-tune transformer model
- [ ] Build production RAG system
- [ ] Multi-modal AI (text + images)

---

## **Resources** 📚

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/
- Claude API: https://docs.anthropic.com/

### Tutorials
- FastAPI + Docker: https://testdriven.io/blog/fastapi-docker/
- AWS ECS: https://docs.aws.amazon.com/ecs/
- RAG Systems: https://www.pinecone.io/learn/rag/

### Communities
- FastAPI Discord: https://discord.gg/fastapi
- MLOps Community: https://mlops.community/
- r/MachineLearning: https://reddit.com/r/MachineLearning

---

## **Troubleshooting** 🔧

### API won't start
```bash
# Check if port is in use
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Models not found
```bash
# Train models first
python train_sms.py

# Check models directory
ls -la ../../../../models/
```

### Docker build fails
```bash
# Clear cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t spam-detection-api:v1.0 .
```

### RAG chatbot API key error
```bash
# Set environment variable
export ANTHROPIC_API_KEY='your-key'

# Or in Windows
set ANTHROPIC_API_KEY=your-key
```

---

**🎉 Congratulations! You now have:**
- ✅ Production-ready REST API
- ✅ Dockerized deployment
- ✅ Understanding of Generative AI (RAG)
- ✅ Foundation for ML Engineering career

**Next: Deploy to cloud and add to your portfolio!**
