# DocuSense — Domain-Specific Document Classifier

An educational NLP web application that classifies documents into **Technology, Healthcare, Finance, Education,** or **Sports**.

## Features
- Paste text or upload TXT, PDF, and DOCX files.
- Text cleanup and document statistics.
- TF-IDF feature extraction with unigrams and bigrams.
- Multi-class Logistic Regression classifier.
- Predicted domain, class probability scores, and important keywords.
- Editable labelled CSV dataset and model retraining.
- Responsive light-theme interface.

## Tech stack
Python, Flask, Pandas, Scikit-learn, TF-IDF, Logistic Regression, HTML, CSS, JavaScript.

## Run on Windows PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## Add training data
Edit `data/training_data.csv`. Keep two columns: `text` and `domain`. Add diverse labelled examples, save, then click **Retrain model** in the app. Keep the five domain labels consistent.

## NLP pipeline
1. Text input / document upload
2. Text extraction (TXT/PDF/DOCX)
3. Basic preprocessing (URL and whitespace cleanup)
4. TF-IDF feature extraction
5. Logistic Regression classification
6. Domain scores, keywords, and document statistics

## Project structure
```text
DocuSense_NLP_Project/
├── app.py
├── requirements.txt
├── Procfile
├── README.md
├── data/training_data.csv
├── templates/index.html
└── static/style.css, app.js
```

**Academic prototype note:** The starter model uses a small hand-curated dataset (60 short examples). Its probabilities are model estimates, not guarantees. Add a larger representative dataset and evaluate on a separate test set before making real-world claims.
