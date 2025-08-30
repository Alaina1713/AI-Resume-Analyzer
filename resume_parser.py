# resume_parser.py
import pdfplumber
import docx2txt
import os

def extract_text_from_pdf(path):
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print("PDF read error:", e)
    return "\n".join(text)

def extract_text_from_docx(path):
    try:
        return docx2txt.process(path) or ""
    except Exception as e:
        print("DOCX read error:", e)
        return ""

def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print("TXT read error:", e)
        return ""

def extract_text(file_path):
    file_path = os.path.normpath(file_path)
    _, ext = os.path.splitext(file_path.lower())

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".doc":
        print("Warning: .doc files are not supported. Please convert to .docx.")
        return ""
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""
