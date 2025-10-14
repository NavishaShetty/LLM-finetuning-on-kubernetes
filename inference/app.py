from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import uvicorn
import time

app = FastAPI()

# Add CORS middleware - CRITICAL for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# microsoft/phi-2 is a large model; using TinyLlama for lighter resource usage
#MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")

MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str
    model: str
    device: str
    generation_time: float

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    start_time = time.time()
    
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_text,
            model=MODEL_NAME,
            device=DEVICE,
            generation_time=generation_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)