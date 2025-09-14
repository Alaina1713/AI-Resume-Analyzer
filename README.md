# AI-Resume-Analyzer

## Overview
AI-Resume-Analyzer is a **full-stack deep learning project** that analyzes resumes, extracts skills, and predicts job-fit scores in real-time.  
It uses **PyTorch and transformer-based models** for NLP and provides a **production-ready API** with a clean, aesthetic frontend.


## Features
- **Deep Learning:** Transformer-based model for multi-label skill prediction
- **Frontend:** Modern, interactive web interface
- **Backend:** Flask API serving PyTorch model
- **Real-Time:** Immediate inference from pasted resume text
- **Extensible:** Can handle PDF resumes, add dashboards, or deploy to cloud/edge
- **Production-Ready:** Clean, modular, and well-documented code


## Tech Stack
- **Backend:** Python, Flask, PyTorch, Transformers
- **Frontend:** HTML, CSS, JavaScript
- **Database:** Optional (JSON/SQLite)
- **Deployment:** Local/Docker/Cloud


## Installation
1. Clone the repository:
```bash
git clone https://github.com/Alaina1713/AI-Resume-Analyzer.git

Create virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt


Start the Flask backend:

cd backend
python app.py


Open your browser and go to http://127.0.0.1:5000/
cd AI-Resume-Analyzer

Usage

Paste your resume text in the textarea

- **Click Analyze Resume** 
- **View job-fit scores for each skill**

## Future Improvements

- **Add PDF parsing for direct resume uploads**
- **Add charts to visualize skill coverage**

Deploy on cloud/edge devices for production
