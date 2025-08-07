from langdetect import detect
from PyPDF2 import PdfReader

LANG_MAP = {
    "es": "spa",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por"
}

def detect_pdf_language(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_sample = ""
        for page in reader.pages[:2]:
            text_sample += page.extract_text() or ""
        if not text_sample.strip():
            return "eng"
        lang_detected = detect(text_sample)
        tesseract_lang = LANG_MAP.get(lang_detected, "eng")
        return tesseract_lang
    except Exception:
        return "eng"
