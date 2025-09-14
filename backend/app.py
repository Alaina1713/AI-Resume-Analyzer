from flask import Flask, request, jsonify, render_template
import torch
from model import ResumeAnalyzer
from preprocess import tokenize

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')

# Load model
model = ResumeAnalyzer(num_labels=10)
model.load_state_dict(torch.load('../models/model_state.pth', map_location='cpu'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['resume_text']
    tokens = tokenize(text)
    with torch.no_grad():
        logits = model(tokens['input_ids'], tokens['attention_mask'])
        preds = torch.sigmoid(logits).squeeze().tolist()
    response = {f'Skill_{i+1}': round(score*100,2) for i, score in enumerate(preds)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
