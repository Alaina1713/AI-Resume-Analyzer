# app.py
import os
import nltk
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from resume_parser import extract_text
from analyzer import (
    compute_similarity_score, skills_match,
    keyword_matching, readability_score,
    formatting_checks, compute_overall_score, top_n_keywords
)

# ensure nltk data is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {'.pdf', '.docx', '.doc', '.txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

@app.route("/", methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=['POST'])
def analyze():
    job_desc = request.form.get("jobdesc", "").strip()
    file = request.files.get("resume")
    if not file or file.filename == "":
        return redirect(url_for('index'))

    fname = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(save_path)

    resume_text = extract_text(save_path)
    if not resume_text:
        resume_text = "(Could not extract text from this file.)"

    # run analyses
    sim_score, tfvec = compute_similarity_score(resume_text, job_desc if job_desc else resume_text)
    found_skills, skills_pct = skills_match(resume_text)
    top_kw, matched_kw, keyword_pct = keyword_matching(job_desc if job_desc else resume_text, resume_text, tfvec)
    read_score = readability_score(resume_text)
    checks, fmt_score = formatting_checks(resume_text)
    overall = compute_overall_score(sim_score, skills_pct, keyword_pct, fmt_score, read_score)

    # prepare results for UI (limit preview)
    resume_preview = resume_text[:4000] + ("..." if len(resume_text) > 4000 else "")
    top_res_keywords = top_n_keywords(resume_text, n=12)

    return render_template("result.html",
                           overall=overall,
                           sim_score=round(sim_score*100, 2),
                           skills_pct=skills_pct,
                           found_skills=found_skills,
                           keyword_pct=keyword_pct,
                           top_kw=top_kw,
                           matched_kw=matched_kw,
                           read_score=read_score,
                           fmt_checks=checks,
                           fmt_score=fmt_score,
                           preview=resume_preview,
                           top_res_keywords=top_res_keywords)

if __name__ == "__main__":
    app.run(debug=True, port=8000)   # you can use 8000, 8080, 7000, etc.

