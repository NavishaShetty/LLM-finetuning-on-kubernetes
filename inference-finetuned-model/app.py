from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import uvicorn
import time

app = FastAPI()

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_MODEL = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "shettynavisha25/tinyllama-alpaca-finetuned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("Loading Fine-tuned TinyLlama Model from HuggingFace")
print("=" * 80)
print(f"Base Model: {BASE_MODEL}")
print(f"Fine-tuned Adapter: {FINETUNED_MODEL}")
print(f"Device: {DEVICE}")
print("=" * 80)

# Load tokenizer
print("\n[1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("✅ Tokenizer loaded")

# Load base model
print("\n[2/3] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
print("✅ Base model loaded")

# Load fine-tuned adapter from HuggingFace
print(f"\n[3/3] Loading fine-tuned adapter from {FINETUNED_MODEL}...")
try:
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    print("✅ Fine-tuned adapter loaded successfully!")
    model_type = "fine-tuned"
except Exception as e:
    print(f"⚠️  Could not load adapter: {e}")
    print("Falling back to base model...")
    model = base_model
    model_type = "base"

model.eval()
print("\n" + "=" * 80)
print(f"Model ready! Type: {model_type}")
print("=" * 80 + "\n")


class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 150
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
        "base_model": BASE_MODEL,
        "finetuned_model": FINETUNED_MODEL,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_type": model_type
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_type": model_type
    }


@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    start_time = time.time()
    
    try:
        # Format prompt for instruction-following (Alpaca format)
        formatted_prompt = f"### Instruction:\n{request.prompt}\n\n### Response:\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in generated_text:
            generated_text = generated_text.split("### Response:")[-1].strip()
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_text,
            model=FINETUNED_MODEL,
            device=DEVICE,
            generation_time=generation_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)