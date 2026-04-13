from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "pidgin-autocomplete-model"

MODEL_NAME = "Ugochief/GPT2_pidgin_autocomplete"

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

model.eval()

class Request(BaseModel):
    text: str

def predict_next_words(prompt, k=5):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    top_k = torch.topk(probs, k)
    return [tokenizer.decode([i]).strip() for i in top_k.indices]

@app.post("/predict")
def predict(req: Request):
    return {"suggestions": predict_next_words(req.text)}