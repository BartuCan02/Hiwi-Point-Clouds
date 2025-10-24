# 🧠 PointLLM Bridge Caption Generator

This repository contains a Python script that automatically generates **structured captions and Q&A conversations** for 3D bridge metadata using **Google Gemini models** (Gemini 2.5 Flash and Gemini 2.5 Pro).

The goal is to produce factual, descriptive, and research-ready captions for 3D point cloud datasets that describe bridges and related structures, including geometry, components, materials, and functional aspects.

---

## 🚀 Features

- Builds detailed natural-language **captions** for given metadata.
- Generates:
  - One descriptive paragraph (`caption`)
  - Three **single-round** Q&A pairs (`single_conversation`)
  - One **multi-round** Q&A chain (`multi_conversation`)
- Uses **Google Gemini API** with automatic token counting, output budgeting, and fallback between models.
- Performs **validation** checks between metadata and model output.
- Outputs results as valid JSON with consistent schema.

---

## 📂 Project Structure
├── generate_captions.py # Main script (your code)
├── metadata_examples.jsonl # Input metadata examples (JSON Lines format)
├── output.json # Example output file
└── README.md 
---

## 🧰 Installation & Setup

### 1. Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

### 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

### 3. Install dependencies
pip install google-generativeai


### 4. Set your Google API key
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
