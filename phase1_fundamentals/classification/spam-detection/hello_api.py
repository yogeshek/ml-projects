from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Hello! I built this API myself!"}

@app.get("/greet/{name}")
def greet(name: str):
    return{f"Hello :{name}"}

@app.get("/add")
def add(a :int, b:int):
    return{f"hi : {a+b}"}
@app.get("/greetings/personal/{name}")
def personal(name: str):
    return(f"hello personal: {name}")

    
#POST
from pydantic import BaseModel # pydantic is a library that validate and structure data

#define the shape of the expected data
class TextInput(BaseModel):
    text: str #required must be text
    reverse: bool = False
    
@app.post("/analyze")
def analyze_text(input: TextInput):
    #count words
    word_count = len(input.text.split())
    
    #char counts
    char_count = len(input.text)
    
    return {
        "original": input.text,
        "word_count": word_count,
        "char_count": char_count
    }

#run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)