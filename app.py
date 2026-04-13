from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

app = FastAPI()

MODEL_PATH = "pidgin-autocomplete-model"

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
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