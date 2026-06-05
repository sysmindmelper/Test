from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import json
from datetime import datetime
import threading
import time
from functools import wraps

# Import your existing modules
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import docx
import io

app = Flask(__name__)
app.secret_key = 'one-two'
CORS(app, supports_credentials=True)
 
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt'}
SESSION_TIMEOUT = 3600 

active_sessions = {}
session_lock = threading.Lock()

# Model configuration
OLLAMA_BASE_URL = "https://ai.live.melp.us/ollama"
AVAILABLE_MODELS = {
    "gpt-oss:20b": "GPT-OSS 20B",
    "deepseek-r1:14b": "DeepSeek R1 14B",
    "qwen3:8b": "Qwen3 8B",
    "llama3.2:latest": "Llama 3.2",
    "llama3.1:latest": "Llama 3.1",
    "mistral:latest": "Mistral",
    "gemma3:4b": "Gemma3 4B"
}

# Session management class
class SessionManager:
    def __init__(self, session_id):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.data = {
            'cv_text': None,
            'jd_text': None,
            'model': None,
            'llm': None,
            'embeddings': None,
            'status': 'initialized',
            'results': {},
            'error': None
        }
    
    def update_access_time(self):
        self.last_accessed = datetime.now()
    
    def is_expired(self):
        return (datetime.now() - self.last_accessed).seconds > SESSION_TIMEOUT

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_or_create_session():
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    with session_lock:
        if session_id not in active_sessions:
            active_sessions[session_id] = SessionManager(session_id)
        else:
            active_sessions[session_id].update_access_time()
    return session_id, active_sessions[session_id]

def require_session(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id, session_mgr = get_or_create_session()
        return f(session_mgr, *args, **kwargs)
    return decorated_function

def cleanup_expired_sessions():
    while True:
        time.sleep(300)
        with session_lock:
            expired = [sid for sid, mgr in active_sessions.items() if mgr.is_expired()]
            for sid in expired:
                del active_sessions[sid]

# Text extraction functions
def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)

def extract_text(file_bytes, ext):
    if ext == '.pdf':
        return extract_text_from_pdf(file_bytes)
    elif ext == '.docx':
        return extract_text_from_docx(file_bytes)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_bytes)
    elif ext == '.txt':
        return file_bytes.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file format: {ext}")


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/session/create', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    with session_lock:
        active_sessions[session_id] = SessionManager(session_id)
    return jsonify({
        'session_id': session_id,
        'created_at': active_sessions[session_id].created_at.isoformat(),
        'status': 'created'
    })

@app.route('/api/models', methods=['GET'])
def get_available_models():
    return jsonify({
        'models': [{'id': k, 'name': v} for k, v in AVAILABLE_MODELS.items()],
        'default': 'llama3.2:latest'
    })

@app.route('/api/session/configure', methods=['POST'])
@require_session
def configure_session(session_mgr):
    data = request.json
    model = data.get('model', 'llama3.2:latest')
    if model not in AVAILABLE_MODELS:
        return jsonify({'error': 'Invalid model selected'}), 400
    try:
        session_mgr.data['model'] = model
        session_mgr.data['llm'] = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=0.1,
            format="json"
        )
        session_mgr.data['embeddings'] = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=model
        )
        session_mgr.data['status'] = 'configured'
        return jsonify({
            'session_id': session_mgr.session_id,
            'model': model,
            'status': 'configured'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/cv', methods=['POST'])
@require_session
def upload_cv(session_mgr):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        try:
            ext = '.' + file.filename.rsplit('.', 1)[1].lower()
            text = extract_text(file.read(), ext)
            session_mgr.data['cv_text'] = text
            return jsonify({
                'session_id': session_mgr.session_id,
                'filename': file.filename,
                'text_length': len(text),
                'status': 'uploaded'
            })
        except Exception as e:
            return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/upload/jd', methods=['POST'])
@require_session
def upload_jd(session_mgr):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        try:
            ext = '.' + file.filename.rsplit('.', 1)[1].lower()
            text = extract_text(file.read(), ext)
            session_mgr.data['jd_text'] = text
            return jsonify({
                'session_id': session_mgr.session_id,
                'filename': file.filename,
                'text_length': len(text),
                'status': 'uploaded'
            })
        except Exception as e:
            return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/process/standardize', methods=['POST'])
@require_session
def standardize_documents(session_mgr):
    if not session_mgr.data.get('cv_text') or not session_mgr.data.get('jd_text'):
        return jsonify({'error': 'Please upload both CV and JD first'}), 400
    if not session_mgr.data.get('llm'):
        return jsonify({'error': 'Please configure model first'}), 400
    try:
        llm = session_mgr.data['llm']
        cv_prompt = PromptTemplate(
            template="""You are an expert HR assistant. Standardize the following CV text into a clean JSON format.
            The JSON should have the following keys: "Work Experience", "Skills", and "Education".
            Extract the relevant information from the text and place it under the appropriate key.
            
            Text: {text}
            
            Return ONLY the JSON object, no additional text."""
        )
        cv_chain = cv_prompt | llm
        standardized_cv = cv_chain.invoke({"text": session_mgr.data['cv_text']})

        jd_prompt = PromptTemplate(
            template="""You are a specialized text formatter. Format the given job description into a structured JSON.
            Extract: 1. Job Title, 2. Job Description, 3. Required Skills, 4. Experience Level, 
            5. Industry Sector, 6. Key Responsibilities
            
            JD: {text}
            
            Return ONLY the JSON object, no additional text."""
        )
        jd_chain = jd_prompt | llm
        standardized_jd = jd_chain.invoke({"text": session_mgr.data['jd_text']})

        session_mgr.data['standardized_cv'] = standardized_cv
        session_mgr.data['standardized_jd'] = standardized_jd

        return jsonify({
            'session_id': session_mgr.session_id,
            'status': 'standardized',
            'cv_standardized': True,
            'jd_standardized': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/round1', methods=['POST'])
@require_session
def generate_round1_questions(session_mgr):
    if not session_mgr.data.get('standardized_cv') or not session_mgr.data.get('standardized_jd'):
        return jsonify({'error': 'Please standardize documents first'}), 400
    try:
        llm = session_mgr.data['llm']
        prompt = """As an expert hiring manager, analyze the CV and JD.
        Identify key similarities and generate exactly 10 interview questions.
        
        Return as JSON with two keys:
        1. "analysis_summary": Brief paragraph of core matches
        2. "interview_questions": List of 10 question strings
        
        CV: {cv}
        JD: {jd}"""

        chain = ChatPromptTemplate.from_template(prompt) | llm
        result = chain.invoke({
            "cv": session_mgr.data['standardized_cv'],
            "jd": session_mgr.data['standardized_jd']
        })

        session_mgr.data['results']['round1'] = result

        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            return jsonify({
                'session_id': session_mgr.session_id,
                'status': 'completed',
                'results': parsed
            })
        except:
            return jsonify({
                'session_id': session_mgr.session_id,
                'status': 'completed',
                'raw_results': result
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/round2/analysis', methods=['POST'])
@require_session
def generate_round2_analysis(session_mgr):
    """Generate Round 2 analysis"""
    if not session_mgr.data.get('standardized_cv') or not session_mgr.data.get('standardized_jd'):
        return jsonify({'error': 'Please standardize documents first'}), 400
    
    try:
        llm = session_mgr.data['llm']
        
        # Analysis
        analysis_prompt = PromptTemplate(
            template="""Compare the CV against the JD and return JSON with:
            1. "matches": List of skills present in both
            2. "gaps": List of missing required skills
            3. "behavioural_questions": 3 behavioral questions
            
            CV: {cv}
            JD: {jd}"""
        )
        analysis_chain = analysis_prompt | llm
        analysis = analysis_chain.invoke({
            "cv": session_mgr.data['standardized_cv'],
            "jd": session_mgr.data['standardized_jd']
        })
        
        # Topics
        topics_prompt = PromptTemplate(
            template="""Identify 5 critical topic areas for interview based on CV and JD.
            Return as comma-separated string.
            
            CV: {cv}
            JD: {jd}"""
        )
        topics_chain = topics_prompt | llm
        topics = topics_chain.invoke({
            "cv": session_mgr.data['standardized_cv'],
            "jd": session_mgr.data['standardized_jd']
        })
        
        session_mgr.data['results']['analysis'] = analysis
        session_mgr.data['results']['topics'] = topics
        
        return jsonify({
            'session_id': session_mgr.session_id,
            'status': 'analyzed',
            'analysis': json.loads(analysis) if isinstance(analysis, str) else analysis,
            'topics': topics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/round2/questions', methods=['POST'])
@require_session
def generate_round2_questions(session_mgr):
    """Generate Round 2 category-specific questions"""
    data = request.json
    category = data.get('category')
    
    categories = {
        'technical': 'Technical Skills Assessment',
        'experience': 'Experience Validation',
        'cultural': 'Cultural Fit & Behavioral',
        'vision': 'Future Vision & Career Growth',
        'domain': 'Domain/Industry Knowledge',
        'problem_solving': 'Problem Solving & Critical Thinking',
        'adaptability': 'Adaptability',
        'communication': 'Communication & Collaboration',
        'leadership': 'Leadership & Initiative'
    }
    
    if category not in categories:
        return jsonify({'error': 'Invalid category'}), 400
    
    if not session_mgr.data.get('results', {}).get('topics'):
        return jsonify({'error': 'Please run Round 2 analysis first'}), 400
    
    try:
        llm = session_mgr.data['llm']
        
        prompt = PromptTemplate(
            template="""Generate interview questions for category: {category}
            
            Context:
            - Topics: {topics}
            - CV: {cv}
            - JD: {jd}
            
            Return JSON array with objects containing:
            - "question": The question
            - "rationale": Why it's relevant
            - "evaluation_criteria": What to listen for
            - "difficulty": Basic/Intermediate/Advanced"""
        )
        
        chain = prompt | llm
        result = chain.invoke({
            "category": categories[category],
            "topics": session_mgr.data['results']['topics'],
            "cv": session_mgr.data['standardized_cv'],
            "jd": session_mgr.data['standardized_jd']
        })
        
        # Store results in memory
        if 'round2_questions' not in session_mgr.data['results']:
            session_mgr.data['results']['round2_questions'] = {}
        session_mgr.data['results']['round2_questions'][category] = result
        
        return jsonify({
            'session_id': session_mgr.session_id,
            'category': category,
            'status': 'generated',
            'questions': json.loads(result) if isinstance(result, str) else result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/status', methods=['GET'])
@require_session
def get_session_status(session_mgr):
    return jsonify({
        'session_id': session_mgr.session_id,
        'status': session_mgr.data['status'],
        'model': session_mgr.data.get('model'),
        'has_cv': bool(session_mgr.data.get('cv_text')),
        'has_jd': bool(session_mgr.data.get('jd_text')),
        'is_standardized': bool(session_mgr.data.get('standardized_cv')),
        'available_results': list(session_mgr.data.get('results', {}).keys()),
        'created_at': session_mgr.created_at.isoformat(),
        'last_accessed': session_mgr.last_accessed.isoformat()
    })

@app.route('/api/results/<result_type>', methods=['GET'])
@require_session
def get_results(session_mgr, result_type):
    mapping = {
        'round1': session_mgr.data['results'].get('round1'),
        'analysis': session_mgr.data['results'].get('analysis'),
        'topics': session_mgr.data['results'].get('topics'),
        'standardized_cv': session_mgr.data.get('standardized_cv'),
        'standardized_jd': session_mgr.data.get('standardized_jd')
    }
    
    # Handle round2_questions with category
    if result_type.startswith('round2_questions_'):
        category = result_type.replace('round2_questions_', '')
        result = session_mgr.data['results'].get('round2_questions', {}).get(category)
    else:
        result = mapping.get(result_type)
    
    if not result:
        return jsonify({'error': 'Result not found'}), 404
    
    # Try to parse JSON if it's a string
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            pass
    
    return jsonify(result)

@app.route('/api/session/cleanup', methods=['DELETE'])
@require_session
def cleanup_session(session_mgr):
    try:
        session_id = session_mgr.session_id
        with session_lock:
            if session_id in active_sessions:
                del active_sessions[session_id]
        return jsonify({'status': 'cleaned', 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
