# 🧠 Open Detector —— Academic Paper AI Detector

<div align="center">

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)

**Language / 语言**: [English](README_EN.md) | [中文](README_ZH.md)

</div>

**Open Detector** is a BERT binary classification model for identifying whether academic papers are closer to **human writing** or **AI writing style**. It serves as an **open alternative** to expensive commercial detectors like Turnitin, suitable for students, individual researchers, and local deployment users.

![Demo](images/demo.png)

> 🌐 **Try Online**: [https://www.xyzscience.com/](https://www.xyzscience.com/)

---

## 🧩 Background & Philosophy

### 💡 Why This Project?

Commercial academic detection services like Turnitin are expensive and not friendly to students, researchers, and self-funded researchers. **Paper detection should be transparent, fair, and explainable, not a commercial black box**. Therefore, we open-source the model to provide the community with a transparent, low-cost solution.

### ⚠️ Philosophy on AI Text Detection

AI's mission is to improve efficiency, not to return people to the era of handwriting.

AI writing far exceeds most human writers in vocabulary selection, syntactic structure, and logical coherence.

❌ **"Writing like AI" ≠ Academic Misconduct.**

Academic integrity should not be judged by language style, but should return to content authenticity.

Judging paper quality solely by style is absurd.

What we should really focus on is not "whether you used AI," but whether the content is authentic, reliable, and free of false generation.

In other words: **We should detect AI hallucinations, not AI writing.**

⚠️ **Note**: This project is currently only **stylometric detection**. The future goal is to build an academic content authenticity detection and AI hallucination identification system.

---

## 🤖 Model Introduction

### ✨ Features

Trained on approximately 1.4 million data samples

- **High Accuracy**: Achieves 99.57% accuracy and 99.58% F1-score on academic text detection
- **Low False Positive Rate**: Only 0.82% false positive rate, minimizing incorrect accusations
- **Exceptional Recall**: 99.94% recall ensures AI-generated content is rarely missed
- **Specialized for Academic Text**: Optimized specifically for academic writing patterns
- **BERT-based Architecture**: Built on BERT-base-uncased for robust semantic understanding

---

## 🎯 Performance

<img src="images/fig1_performance_comparison.png" width="600" alt="Performance Comparison">

<img src="images/fig2_confusion_matrix.png" width="500" alt="Confusion Matrix">

---

## 🚀 Quick Start

Model files are available at [https://huggingface.co/followsci/bert-ai-text-detector](https://huggingface.co/followsci/bert-ai-text-detector)

### Install

```bash
pip install transformers torch
```

### Run

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "followsci/bert-ai-text-detector"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "Your academic paragraph here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_prob = probs[0][1].item() * 100
    human_prob = probs[0][0].item() * 100
    
    print(f"AI-generated: {ai_prob:.1f}% | Human: {human_prob:.1f}%")
```

### Parameters

- **Labels**: `0 = Human`, `1 = AI`
- **Max Length**: 512 tokens

---

## 🚧 What We're Working On

### Research Directions

| Future Direction | Description |
|-----------------|-------------|
| Fact Consistency Verification | Citation chain checking, literature search comparison |
| AI Hallucination Detection | Focus on distinguishing real vs. fabricated content |
| Citation Authenticity | Prevent "fake citations" and "model-generated references" |
| Academic Logic Consistency | Structure and reasoning verification |
| Multi-language Extension | Support Chinese, Japanese, and other languages |

### Ultimate Goal

Build a framework for "AI-assisted authentic academia," not an "anti-AI writing" tool.

---

## ✨ About Humanization Rewriting Model

We have also trained an academic paper humanization rewriting model:

- Maintains academic expression style
- Eliminates AI writing traces
- Avoids misjudgment by style detection

📌 This model can be used for free at [https://www.xyzscience.com/](https://www.xyzscience.com/).

---

## 📌 Final Thoughts

> **AI should not be judged, but should become a tool to support knowledge creation.**  
> **Our goal is not to punish AI, but to protect academic authenticity.**

Thank you for reading. Welcome to Star ⭐ to support the open academic tools ecosystem.

