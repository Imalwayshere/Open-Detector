"""
Simple example for using the AI-Generated Text Detection Model

Model: followsci/bert-ai-text-detector
Available at: https://huggingface.co/followsci/bert-ai-text-detector
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification


# Load model and tokenizer
print("Loading model...")
model_name = "followsci/bert-ai-text-detector"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()
print("Model loaded!")

# Example 1: Single text detection
text = "Your academic paragraph here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_prob = probs[0][1].item() * 100
    human_prob = probs[0][0].item() * 100

print(f"\nAI-generated: {ai_prob:.1f}%")
print(f"Human-written: {human_prob:.1f}%")

if ai_prob > 50:
    print("Prediction: AI-generated")
else:
    print("Prediction: Human-written")

# Example 2: Batch processing
print("\n" + "="*50)
print("Batch Processing Example")
print("="*50)

texts = [
    "First paragraph text here...",
    "Second paragraph text here...",
    # Add more texts
]

inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

for i, prob in enumerate(probs):
    ai_prob = prob[1].item() * 100
    print(f"\nText {i+1}: AI probability = {ai_prob:.1f}%")
