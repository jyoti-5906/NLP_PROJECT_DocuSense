from flask import Flask, render_template, request, jsonify
from pathlib import Path
import io, re
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'training_data.csv'
MODEL = BASE / 'data' / 'classifier.joblib'
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

SAMPLES = {
'Technology': [
'Software developers build applications using Python Java APIs databases and cloud computing.',
'Artificial intelligence machine learning neural networks and deep learning are changing technology.',
'Cybersecurity teams protect computer networks servers systems and user data from malware.',
'Developers use Git version control programming frameworks and software testing tools.',
'Cloud infrastructure provides scalable storage virtual machines and deployment services.',
'A mobile application uses an interface backend database and authentication system.',
'Data science uses algorithms Python programming analytics and machine learning models.',
'The computer network uses routers protocols bandwidth and secure connections.',
'The software engineering team fixed bugs and improved application performance.',
'Robotics combines sensors automation computer vision and intelligent control systems.',
'The website uses HTML CSS JavaScript and a Python server to process requests.',
'The processor architecture improves computing speed memory and system performance.'],
'Healthcare': [
'Doctors diagnose patients and recommend treatment based on medical history and symptoms.',
'Hospitals provide emergency care surgery nursing and clinical laboratory services.',
'Public health research studies disease prevention vaccination and community health.',
'The patient received a prescription after a medical examination and blood test.',
'Healthcare professionals monitor blood pressure heart rate and other vital signs.',
'Clinical trials evaluate the safety and effectiveness of a new medicine.',
'The hospital introduced digital medical records for patient diagnosis and treatment.',
'Nutrition exercise and regular screening can help prevent chronic diseases.',
'Researchers studied infection rates and the spread of respiratory illness.',
'A physician reviewed the scan results and discussed the treatment plan.',
'Mental health services provide counselling and support for emotional wellbeing.',
'The laboratory analysed tissue samples to support an accurate diagnosis.'],
'Finance': [
'The bank reported quarterly revenue profit and earnings for its investors.',
'Stock market prices changed after the central bank announced interest rates.',
'Financial analysts evaluate assets liabilities cash flow and company valuation.',
'The investment portfolio includes bonds equities mutual funds and government securities.',
'Inflation affects consumer spending purchasing power and monetary policy.',
'The company prepared its annual balance sheet and income statement.',
'A loan application is assessed using credit score income and repayment history.',
'The insurance policy covers financial losses subject to the contract terms.',
'Investors compare market risk returns and dividend performance.',
'The accountant reviewed expenses taxation invoices and financial statements.',
'The central bank uses monetary policy to manage liquidity and inflation.',
'The startup raised capital through venture funding and private investors.'],
'Education': [
'Students attend lectures complete assignments and prepare for university examinations.',
'Teachers design lesson plans and assess student learning outcomes in classrooms.',
'The university offers undergraduate postgraduate and research degree programmes.',
'Online learning platforms provide courses quizzes and digital study materials.',
'The curriculum includes mathematics science literature and practical activities.',
'Academic researchers publish papers and present findings at education conferences.',
'The school introduced a new assessment method to evaluate student progress.',
'Libraries provide textbooks journals and other resources for learners.',
'A scholarship supports students with tuition fees and educational expenses.',
'The professor explained the concept and guided students through a laboratory exercise.',
'Educational technology supports interactive lessons and remote classroom participation.',
'The student submitted a project report as part of the semester coursework.'],
'Sports': [
'The football team won the match after scoring two goals in the final minutes.',
'The cricket player scored a century during the international tournament.',
'Athletes train daily to improve speed strength stamina and performance.',
'The coach announced the squad for the upcoming championship.',
'The basketball game featured rebounds assists and three point shots.',
'The tennis player advanced to the final after winning the semifinal.',
'Fans watched the tournament as teams competed for the league title.',
'The runner broke the national record in the athletics competition.',
'The referee awarded a penalty after a foul inside the box.',
'The team prepared its strategy for the cricket world cup.',
'Swimming competitions require endurance technique and disciplined training.',
'The player recovered from an injury and returned to professional competition.']}

def ensure_data():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        pd.DataFrame([{'text': t, 'domain': d} for d, arr in SAMPLES.items() for t in arr]).to_csv(DATA, index=False)

def train():
    ensure_data()
    df = pd.read_csv(DATA).dropna()
    pipe = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=12000, sublinear_tf=True)), ('clf', LogisticRegression(max_iter=1500, class_weight='balanced'))])
    pipe.fit(df.text.astype(str), df.domain.astype(str))
    joblib.dump(pipe, MODEL)
    return pipe, len(df)

ensure_data()
if MODEL.exists():
    model = joblib.load(MODEL)
    training_count = len(pd.read_csv(DATA))
else:
    model, training_count = train()

def get_text(file):
    raw = file.read()
    name = file.filename.lower()
    if name.endswith('.txt'): return raw.decode('utf-8', errors='ignore')
    if name.endswith('.pdf'): return '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(raw)).pages)
    if name.endswith('.docx'): return '\n'.join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
    raise ValueError('Please upload a TXT, PDF, or DOCX file.')

def classify(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = re.findall(r"\b[\w'-]+\b", text)
    if len(words) < 8: raise ValueError('Please enter a document with at least 8 words.')
    probs = model.predict_proba([text])[0]
    labels = model.named_steps['clf'].classes_
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    vec = model.named_steps['tfidf'].transform([text]).toarray()[0]
    terms = model.named_steps['tfidf'].get_feature_names_out()
    keys = [{'word': str(terms[i]), 'score': round(float(vec[i]), 3)} for i in vec.argsort()[::-1] if vec[i] > 0][:8]
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    return {'domain': ranked[0][0], 'confidence': round(float(ranked[0][1])*100,1), 'scores':[{'label':a,'score':round(float(b)*100,1)} for a,b in ranked], 'keywords':keys, 'stats':{'words':len(words),'sentences':len(sentences),'characters':len(text),'stop_words':sum(w.lower() in ENGLISH_STOP_WORDS for w in words)}}

@app.route('/')
def home(): return render_template('index.html', training_count=training_count)
@app.route('/classify', methods=['POST'])
def classify_route():
    try:
        f=request.files.get('document')
        text=get_text(f) if f and f.filename else request.form.get('text','')
        return jsonify({'success':True, **classify(text)})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),400
@app.route('/retrain', methods=['POST'])
def retrain():
    global model, training_count
    try:
        model, training_count=train()
        return jsonify({'success':True,'message':f'Model retrained with {training_count} labelled examples.'})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.errorhandler(413)
def too_large(_): return jsonify({'success':False,'error':'Maximum file size is 8 MB.'}),413
if __name__ == '__main__': app.run(debug=True)
