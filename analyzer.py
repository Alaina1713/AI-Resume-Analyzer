# analyzer.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import textstat

# Download only once
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOP = set(stopwords.words("english"))

# Basic skills list - expand this as needed
COMMON_SKILLS = [
    "python", "flask", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "keras",
    "nlp", "natural language processing", "machine learning", "deep learning",
    "docker", "git", "linux", "aws", "excel", "power bi", "matplotlib", "seaborn",
    "javascript", "html", "css", "postgresql", "mysql", "opencv", "pytorch",
    "spacy", "textblob", "nltk", "transformer", "transformers", "api", "rest"
]

def simple_tokenize(text):
    """Tokenize into words, lowercase, remove stopwords & short tokens"""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = wordpunct_tokenize(text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 1]

def top_n_keywords(text, n=12):
    toks = simple_tokenize(text)
    cnt = Counter(toks)
    return [w for w,_ in cnt.most_common(n)]

def compute_similarity_score(resume_text, job_text):
    """Compute cosine similarity between resume and job description"""
    tf = TfidfVectorizer(stop_words="english")
    docs = [job_text, resume_text]
    try:
        vecs = tf.fit_transform(docs)
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
    except Exception:
        sim = 0.0
    return float(sim), tf

def skills_match(resume_text, skills_list=COMMON_SKILLS):
    resume_tokens = set(simple_tokenize(resume_text))
    found = [s for s in skills_list if s.lower() in resume_tokens]
    pct = (len(found) / len(skills_list)) * 100 if skills_list else 0
    return found, round(pct, 2)

def keyword_matching(job_text, resume_text, tf_vectorizer=None, top_k=10):
    job_tokens = simple_tokenize(job_text)
    resume_tokens = set(simple_tokenize(resume_text))
    job_counter = Counter(job_tokens)
    top_job = [k for k,_ in job_counter.most_common(top_k)]
    matched = [k for k in top_job if k in resume_tokens]
    percent = (len(matched) / len(top_job)) * 100 if top_job else 0
    return top_job, matched, round(percent, 2)

def readability_score(resume_text):
    try:
        flesch = textstat.flesch_reading_ease(resume_text)
    except Exception:
        flesch = 0
    flesch = max(0, min(100, flesch))  # clamp to [0,100]
    return round(flesch, 2)

def formatting_checks(resume_text):
    txt = resume_text.lower()
    checks = {
        "has_education": "education" in txt,
        "has_experience": "experience" in txt or "work" in txt,
        "has_contact": bool(re.search(r"@\w+\.\w+", resume_text)) or bool(re.search(r"\+?\d{7,}", resume_text)),
        "has_skills_section": "skill" in txt,
    }
    score = sum(1 for v in checks.values() if v) / len(checks) * 100
    return checks, round(score, 2)

def compute_overall_score(similarity, skills_pct, keyword_pct, formatting_pct, readability):
    sim_w = similarity * 100  # similarity is 0–1, normalize to %
    score = (sim_w * 0.40) + (skills_pct * 0.30) + (keyword_pct * 0.15) + (formatting_pct * 0.10) + (readability * 0.05)
    return round(score, 2)
