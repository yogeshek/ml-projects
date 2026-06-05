# 🌱 First Steps: From ML Model to Web API

**Start here if you only know Python and ML basics.**

This guide assumes you know:
- ✅ Python programming
- ✅ ML models (train, predict)
- ✅ pandas, scikit-learn

This guide teaches you:
- ❓ What is a web API?
- ❓ How to make your code accessible over internet
- ❓ How to build your first API

**Time commitment:** 1 hour today, then 30 mins/day for 7 days.

---

## 📍 **Day 0: Understanding APIs (Today - 1 hour)**

### **What You Need to Know First**

#### **Your Current Setup:**
```
You run on YOUR computer:
┌──────────────────────────┐
│  python train_sms.py     │  → Trains model
│  python predict_sms.py   │  → Makes predictions
└──────────────────────────┘

Problem: Only YOU can use it, only on YOUR computer
```

#### **With an API:**
```
Anyone, anywhere:
┌─────────────────────────┐
│  Your Friend's Phone    │
│  Your Laptop            │
│  Another Program        │
└───────────┬─────────────┘
            │ Sends message over internet
            ↓
┌─────────────────────────┐
│  Your API (Running)     │
│  - Receives message     │
│  - Runs your ML model   │
│  - Sends back answer    │
└─────────────────────────┘

Result: Everyone can use your model!
```

#### **Real-World Examples You Use Daily:**

1. **Weather App on Phone:**
   - App sends your location to Weather API
   - Weather API sends back temperature
   - You see: "25°C, Sunny"

2. **Google Maps:**
   - You type destination
   - App calls Google Maps API
   - Gets back directions

3. **Online Shopping:**
   - You click "Buy"
   - Website calls Payment API
   - Payment processed

**An API is just a way to let other programs use your code!**

---

## 🎯 **Task 1: See Your API Working (10 minutes)**

The API is already running on your computer. Let's just look at it!

### **Step 1: Open in Browser**

1. Open your web browser (Chrome, Firefox, etc.)
2. Go to: **http://localhost:8000/docs**
3. You'll see a web page with green/blue boxes

**What you're looking at:** Your API's "menu" - all the things it can do!

### **Step 2: Test a Prediction**

1. Find the green box that says **POST /predict**
2. Click **"Try it out"** button
3. You'll see a text box with JSON code
4. Replace it with this:
   ```json
   {
     "message": "FREE money! Click here NOW!!!"
   }
   ```
5. Click the blue **"Execute"** button
6. Scroll down to see the response

**You should see:**
```json
{
  "success": true,
  "final_verdict": "SPAM",
  "confidence": 98.5,
  ...
}
```

**What just happened?**
- You sent a message to your API
- Your API ran it through your 4 trained models
- Your API sent back: "This is SPAM"
- **That's what APIs do!**

### **Step 3: Try Another Message**

Do it again with:
```json
{
  "message": "Hey, are you free for dinner tomorrow?"
}
```

**You should see:** `"final_verdict": "HAM"`

**Congratulations! You just used a web API!** 🎉

---

## 💻 **Task 2: Use API from Command Line (10 minutes)**

APIs aren't just for browsers. Let's call it from the terminal:

### **Open your terminal and run:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "Win FREE iPhone!"}'
```

**You'll see JSON response printed in terminal!**

**What does this command mean?**
- `curl` = a tool to send web requests
- `-X POST` = sending data (POST method)
- `http://localhost:8000/predict` = where to send it
- `-d '{"message": "..."}'` = the data you're sending

**This is how other programs will call your API!**

---

## 🔍 **Task 3: Understand How It Works (15 minutes)**

Open the file `api.py` in your editor. Don't worry about understanding everything! Just find these sections:

### **Section 1: Create the API (around line 18)**
```python
app = FastAPI(
    title="SMS Spam Detection API",
    description="Multi-model spam detection system",
    version="1.0.0"
)
```
**This creates your API. Like creating a Flask app if you know Flask.**

### **Section 2: The Prediction Endpoint (around line 230)**
```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict if a message is spam or ham"""
    
    # Your code does the same 4 steps as your training script:
    # 1. Clean the text
    # 2. Vectorize it
    # 3. Run through models
    # 4. Return prediction
```

**Compare to your `predict_sms.py` - it's doing the SAME thing!**

The only differences:
- Instead of reading from file → receives data from HTTP request
- Instead of printing to console → returns JSON

**That's the entire concept of an API!**

---

## 📚 **Task 4: Learn the Basics (25 minutes)**

### **Watch This Video (10 minutes):**
**"What is a REST API?"**
https://www.youtube.com/watch?v=lsMQRaeKNDk

### **Read This Article (15 minutes):**
**FastAPI Tutorial - First Steps**
https://fastapi.tiangolo.com/tutorial/first-steps/

Just read the "First Steps" page. Don't code yet, just understand the concepts.

---

## ✅ **Day 0 Success Criteria**

By end of today, you should be able to explain:

- [ ] What is an API in simple words?
- [ ] What is the difference between running a Python script and running an API?
- [ ] What is localhost:8000?
- [ ] What is JSON?
- [ ] What happens when you send a POST request to /predict?

**If you can explain these to yourself (or someone else), you're ready for Day 1!**

---

## 🚀 **Day 1: Your First API (Tomorrow - 30 minutes)**

### **Goal:** Build the simplest possible API from scratch

### **Task: Create Hello World API**

1. **Create new file:** `hello_api.py`

2. **Type this code** (don't copy-paste! typing helps you learn):

```python
from fastapi import FastAPI

# Create the API
app = FastAPI()

# Define a route
@app.get("/")
def home():
    return {"message": "Hello! I built this API myself!"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"greeting": f"Hello {name}!"}

@app.get("/add")
def add(a: int, b: int):
    return {"result": a + b}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

3. **Install FastAPI if needed:**
```bash
pip install fastapi uvicorn
```

4. **Run it:**
```bash
python hello_api.py
```

5. **Test it:**
   - Browser: http://localhost:8001/
   - Browser: http://localhost:8001/greet/Yogesh
   - Browser: http://localhost:8001/add?a=5&b=3
   - Docs: http://localhost:8001/docs

### **Understanding Each Part:**

```python
@app.get("/")
```
- `@app.get` = This endpoint responds to GET requests
- `"/"` = The URL path (root)

```python
def home():
    return {"message": "Hello!"}
```
- Function name doesn't matter (can be anything)
- Returns a dictionary → FastAPI converts to JSON automatically

```python
@app.get("/greet/{name}")
def greet(name: str):
```
- `{name}` = path parameter (like a variable in the URL)
- `name: str` = FastAPI validates it's a string

```python
@app.get("/add")
def add(a: int, b: int):
```
- `?a=5&b=3` = query parameters
- FastAPI automatically converts to integers

### **Experiments to Try:**

1. Add a new endpoint `/goodbye/{name}` that returns goodbye message
2. Add `/multiply?x=4&y=5` endpoint
3. What happens if you go to a URL that doesn't exist?
4. What happens if you pass text instead of number to `/add`?

### **Day 1 Success:**
- [ ] I created and ran my own API
- [ ] I understand @app.get
- [ ] I understand path parameters
- [ ] I understand query parameters
- [ ] I tested all endpoints

---

## 📊 **Day 2: Add POST Endpoint (30 minutes)**

### **Goal:** Learn to receive data in request body

### **Modify your hello_api.py:**

Add this at the top:
```python
from pydantic import BaseModel
```

Add this class (before your endpoints):
```python
class TextInput(BaseModel):
    text: str
    reverse: bool = False
```

Add this new endpoint:
```python
@app.post("/analyze")
def analyze_text(input: TextInput):
    # Count words
    word_count = len(input.text.split())
    
    # Count characters
    char_count = len(input.text)
    
    # Reverse if requested
    result_text = input.text[::-1] if input.reverse else input.text
    
    return {
        "original": input.text,
        "processed": result_text,
        "word_count": word_count,
        "char_count": char_count
    }
```

### **Test it:**

1. Restart your API: `python hello_api.py`
2. Go to: http://localhost:8001/docs
3. Find POST /analyze
4. Click "Try it out"
5. Enter:
```json
{
  "text": "Hello World",
  "reverse": true
}
```
6. Click Execute

### **Understanding:**

```python
class TextInput(BaseModel):
    text: str
    reverse: bool = False
```
- This defines the **shape** of data you expect
- `text: str` = required field, must be string
- `reverse: bool = False` = optional field, defaults to False

```python
def analyze_text(input: TextInput):
```
- FastAPI automatically validates incoming data
- If data doesn't match TextInput, returns error
- You get a nice Python object to work with

### **Day 2 Success:**
- [ ] I understand GET vs POST
- [ ] I understand request body
- [ ] I understand Pydantic models
- [ ] I can send JSON data to API

---

## 🤖 **Day 3: Add Your ML Model (1 hour)**

### **Goal:** Connect ONE of your trained models to an API

### **Create new file:** `my_spam_api.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

# Create API
app = FastAPI(title="My Spam API", version="1.0.0")

# Global variables for model
model = None
vectorizer = None

# Request model
class Message(BaseModel):
    text: str

# Load models when API starts
@app.on_event("startup")
def load_models():
    global model, vectorizer
    
    try:
        # Update this path to where your models are
        models_path = Path(__file__).parent.parent.parent.parent / "ml-projects/models"
        
        print(f"Loading models from: {models_path}")
        
        # Load vectorizer
        vectorizer = joblib.load(models_path / "vectorizer.pkl")
        print("✓ Vectorizer loaded")
        
        # Load just Naive Bayes model (start simple!)
        model_data = joblib.load(models_path / "naive_bayes.pkl")
        model = model_data['model']
        accuracy = model_data['test_accuracy']
        print(f"✓ Model loaded (Accuracy: {accuracy*100:.1f}%)")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise

# Clean text (simplified version from your training)
def clean_text(text: str) -> str:
    # For now, just lowercase and basic cleaning
    # You can add your full preprocessing later
    return text.lower().strip()

# Health check
@app.get("/")
def home():
    return {
        "message": "My Spam Detection API",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

# Prediction endpoint
@app.post("/predict")
def predict(msg: Message):
    # Check if model is loaded
    if not model or not vectorizer:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    try:
        # Step 1: Clean text
        cleaned = clean_text(msg.text)
        
        # Step 2: Vectorize (same as training!)
        vectorized = vectorizer.transform([cleaned])
        
        # Step 3: Predict
        prediction = model.predict(vectorized)[0]
        
        # Step 4: Get probability
        probabilities = model.predict_proba(vectorized)[0]
        
        # Step 5: Format result
        return {
            "message": msg.text,
            "cleaned": cleaned,
            "prediction": "spam" if prediction == 1 else "ham",
            "confidence": float(probabilities[prediction] * 100),
            "probabilities": {
                "ham": float(probabilities[0] * 100),
                "spam": float(probabilities[1] * 100)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### **Run it:**

```bash
python my_spam_api.py
```

**Watch the startup logs:**
```
✓ Vectorizer loaded
✓ Model loaded (Accuracy: 97.2%)
```

### **Test it:**

```bash
# Health check
curl http://localhost:8002/health

# Test with spam
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE money! Click NOW!"}'

# Test with ham
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hey, want to grab coffee?"}'
```

### **Understanding New Concepts:**

```python
@app.on_event("startup")
def load_models():
```
- Runs ONCE when API starts
- Perfect for loading models (slow operation)
- Models stay in memory, ready for predictions

```python
global model, vectorizer
```
- Makes variables accessible across all functions
- Load once, use many times = fast predictions!

```python
raise HTTPException(status_code=503, ...)
```
- Returns proper HTTP error codes
- 503 = Service Unavailable (model not loaded)
- 500 = Internal Server Error (prediction failed)

### **Day 3 Success:**
- [ ] I loaded my ML model in FastAPI
- [ ] I can make predictions via API
- [ ] I understand @app.on_event("startup")
- [ ] My API returns proper JSON responses
- [ ] I tested with both spam and ham messages

---

## 🎯 **Day 4: Add All 4 Models (30 minutes)**

### **Goal:** Use all your models with ensemble voting

### **Modify your my_spam_api.py:**

Change the global variables:
```python
models = {}  # Dictionary to hold all models
vectorizer = None
```

Update load_models():
```python
@app.on_event("startup")
def load_models():
    global models, vectorizer
    
    try:
        models_path = Path(__file__).parent.parent.parent.parent / "ml-projects/models"
        
        # Load vectorizer
        vectorizer = joblib.load(models_path / "vectorizer.pkl")
        print("✓ Vectorizer loaded")
        
        # Load all 4 models
        model_names = ['naive_bayes', 'random_forest', 
                      'logistic_regression', 'svc']
        
        for name in model_names:
            model_data = joblib.load(models_path / f"{name}.pkl")
            models[name] = model_data
            acc = model_data['test_accuracy']
            print(f"✓ {name}: {acc*100:.1f}% accuracy")
        
        print(f"\n✓ Loaded {len(models)} models successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
```

Add a new endpoint for ensemble prediction:
```python
@app.post("/predict_ensemble")
def predict_ensemble(msg: Message):
    if not models or not vectorizer:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Clean and vectorize
        cleaned = clean_text(msg.text)
        vectorized = vectorizer.transform([cleaned])
        
        # Get predictions from all models
        predictions = {}
        votes = {"spam": 0, "ham": 0}
        
        for name, model_data in models.items():
            model = model_data['model']
            
            # Predict
            pred = model.predict(vectorized)[0]
            pred_label = "spam" if pred == 1 else "ham"
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(vectorized)[0][1] * 100
            else:
                prob = None
            
            predictions[name] = {
                "prediction": pred_label,
                "confidence": prob,
                "model_accuracy": model_data['test_accuracy'] * 100
            }
            
            # Count votes
            votes[pred_label] += 1
        
        # Determine final verdict
        final_verdict = "spam" if votes["spam"] > votes["ham"] else "ham"
        
        return {
            "message": msg.text,
            "cleaned": cleaned,
            "predictions": predictions,
            "votes": votes,
            "final_verdict": final_verdict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### **Test ensemble:**

```bash
curl -X POST "http://localhost:8002/predict_ensemble" \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! Claim your $1000 prize NOW!!!"}'
```

### **Day 4 Success:**
- [ ] All 4 models loaded
- [ ] Ensemble voting works
- [ ] I understand majority voting
- [ ] I can see all model predictions

---

## 📝 **Day 5-7: Practice & Experiments**

### **Day 5: Add Features**

Add these to your API:
- [ ] `/models` endpoint that returns info about all models
- [ ] Better text cleaning (use your full preprocessing)
- [ ] Request validation (max message length)

### **Day 6: Error Handling**

Improve error handling:
- [ ] What if message is empty?
- [ ] What if message is too long?
- [ ] Add helpful error messages

### **Day 7: Documentation**

Make your API professional:
- [ ] Add descriptions to all endpoints
- [ ] Add examples in docstrings
- [ ] Test everything in Swagger UI

---

## 🎉 **Week 1 Complete!**

**By end of Week 1, you should have:**

✅ A working understanding of web APIs  
✅ Your own FastAPI application  
✅ Your ML models served via API  
✅ Ensemble prediction system  
✅ Proper error handling  
✅ Good documentation  

**You can now say:** "I built a production-ready ML API!"

---

## 🚫 **What NOT to Worry About Yet**

### **Don't Learn These Yet (Week 2+):**

- ❌ Docker (Week 2)
- ❌ Deployment/Cloud (Week 3)
- ❌ Databases
- ❌ Authentication
- ❌ Load balancing
- ❌ Kubernetes
- ❌ CI/CD pipelines

**Focus ONLY on understanding APIs this week!**

---

## 🤔 **Common Beginner Questions**

### **Q: What is localhost?**
A: Your own computer. `localhost` = `127.0.0.1` = "this computer"

### **Q: What is a port (like :8000)?**
A: Like an apartment number. Your computer has 65,535 "doors". Port 8000 is door #8000.

### **Q: Why use FastAPI instead of Flask?**
A: 
- Faster
- Automatic documentation
- Type checking
- Async support
- More modern

### **Q: What is JSON?**
A: A text format for data. Like a dictionary but as text:
```python
# Python dictionary
{"name": "John", "age": 30}

# JSON (looks the same!)
{"name": "John", "age": 30}
```

### **Q: What's the difference between GET and POST?**
A:
- **GET**: "Get me some data" (reading)
  - Example: Get user profile
  - Data in URL: `/user?id=123`
  
- **POST**: "I'm sending you data" (writing)
  - Example: Submit a form
  - Data in body: `{"name": "John"}`

### **Q: Do I need to know HTTP in detail?**
A: Not yet! Just know:
- 200 = Success
- 404 = Not found
- 500 = Server error

### **Q: Can others access my localhost:8000?**
A: No! Only you. Localhost = only your computer. To share, you need to deploy to cloud (Week 3).

---

## 📚 **Essential Resources**

### **Bookmark These:**

1. **FastAPI Documentation**
   - https://fastapi.tiangolo.com/
   - Best API framework docs ever written!

2. **HTTP Status Codes**
   - https://httpstatuses.com/
   - Quick reference

3. **JSON Formatter**
   - https://jsonformatter.org/
   - Paste JSON, see it formatted

### **Keep These Handy:**

```bash
# Start your API
python my_spam_api.py

# Test with curl
curl http://localhost:8002/health

# POST request
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "your message"}'

# Check if port is in use (Windows)
netstat -ano | findstr :8002

# Kill process (Windows)
taskkill /PID <process_id> /F
```

---

## ✅ **Daily Progress Tracker**

```markdown
### Week 1: API Basics

Day 0: □ Tested existing API
       □ Understood what APIs are
       □ Watched REST API video
       
Day 1: □ Built hello_api.py
       □ Tested GET endpoints
       □ Understood path/query params
       
Day 2: □ Added POST endpoint
       □ Learned Pydantic models
       □ Tested request body
       
Day 3: □ Loaded ML model
       □ Made first prediction via API
       □ Understood @app.on_event
       
Day 4: □ Loaded all 4 models
       □ Implemented ensemble voting
       □ Tested majority voting
       
Day 5: □ Added features
       □ Improved preprocessing
       □ Added validation
       
Day 6: □ Better error handling
       □ Tested edge cases
       □ Added helpful messages
       
Day 7: □ Documented everything
       □ Tested in Swagger UI
       □ Week 1 review
```

---

## 🎯 **Your Success Metrics**

**By end of this guide, you should be able to:**

1. **Explain to a friend:**
   - What is a web API?
   - How does your spam detector work as an API?
   - What is the difference between GET and POST?

2. **Build from scratch:**
   - A simple FastAPI app
   - Endpoints with path/query parameters
   - POST endpoints with request body
   - Load and serve ML models

3. **Debug issues:**
   - Read error messages
   - Fix common problems
   - Test endpoints

4. **Show off:**
   - Working API on localhost
   - Swagger documentation
   - Real predictions via HTTP

---

## 🚀 **What's Next? (Week 2)**

**After completing Week 1, you'll learn:**

1. **Docker Basics** (3 days)
   - What are containers?
   - Why use Docker?
   - Run your API in a container

2. **Deploy Locally** (2 days)
   - Docker Compose
   - Multiple services
   - Networking

3. **Polish** (2 days)
   - Logging
   - Monitoring
   - Testing

**But don't think about Week 2 yet! Focus on mastering Week 1 first.**

---

## 💡 **Final Tips**

### **Do's:**
✅ Type code yourself (don't copy-paste)  
✅ Break things and fix them  
✅ Test after every small change  
✅ Read error messages carefully  
✅ Use Swagger UI extensively  
✅ Ask questions when stuck  

### **Don'ts:**
❌ Don't rush through  
❌ Don't skip the basics  
❌ Don't try to learn everything at once  
❌ Don't compare to senior engineers  
❌ Don't worry about "perfect" code  

---

## 🆘 **Getting Help**

### **When Stuck:**

1. **Read the error message** - It usually tells you what's wrong
2. **Google the exact error** - Someone had this problem before
3. **Check FastAPI docs** - They're excellent!
4. **Ask specific questions:**
   - ✅ "I get error X when doing Y, what does it mean?"
   - ✅ "Why use POST here instead of GET?"
   - ❌ "It doesn't work" (too vague)
   - ❌ "Write the code for me"

### **Good Question Format:**
```markdown
**What I'm trying to do:** Load my model in FastAPI

**What happens:** Error message about file not found

**Error message:** 
FileNotFoundError: [Errno 2] No such file or directory: 'model.pkl'

**My code:**
model = joblib.load("model.pkl")

**What I've tried:**
- Checked if file exists
- Tried absolute path
```

---

## 🎊 **Celebration Milestones**

### **Celebrate when you:**

- ✅ Successfully run hello_api.py
- ✅ See "Hello Yogesh!" in your browser
- ✅ Successfully POST data
- ✅ Load your first model in API
- ✅ Get your first spam prediction via API
- ✅ See all 4 models voting together
- ✅ Complete Week 1!

**Each milestone matters! You're building real engineering skills!**

---

## 🎓 **You're Ready When...**

**You'll know you're ready for Week 2 (Docker) when:**

- [ ] You can build a FastAPI app from scratch
- [ ] You understand GET vs POST
- [ ] You can load and use your ML models
- [ ] You can test your API in browser and curl
- [ ] You can explain what an API is to someone else
- [ ] You feel comfortable with the concepts

**If you can do these, you're crushing it! Move to Week 2!**

**If not, that's OK! Review, practice, ask questions. Speed doesn't matter - understanding does.**

---

## 📖 **Next Documents to Read**

**After completing this guide:**

1. **LEARNING_PATH.md** - Detailed 10-day curriculum (Week 2+)
2. **VISUAL_GUIDE.md** - Visual diagrams and architecture
3. **DEPLOYMENT_GUIDE.md** - Docker and cloud deployment (Week 3)

**But finish Week 1 first! One step at a time.**

---

**Ready to start? Your first task is just 10 minutes away:**

Go to http://localhost:8000/docs and click around. That's it!

**You've got this! 🚀**

---

*Created: 2026-06-05*  
*For: Beginners who only know Python + ML*  
*Next: LEARNING_PATH.md (after Week 1)*
