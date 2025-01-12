from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
from docx import Document
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
import gc

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

DOCUMENT_PATH = os.getenv('DOCUMENT_PATH', 'arabic_file.docx')
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

app = Flask(__name__)
CORS(app, 
    resources={
        r"/api/*": {
            "origins": ["https://superlative-belekoy-1319b4.netlify.app"],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "expose_headers": ["Access-Control-Allow-Origin"],
            "supports_credentials": True
        }
    })

def preprocess_arabic_text(text):
    """Preprocess Arabic text for better matching"""
    if not text:
        return ""
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variations
    text = re.sub('[إأٱآا]', 'ا', text)
    # Normalize teh marbuta and ha
    text = text.replace('ة', 'ه')
    # Normalize alef maksura and ya
    text = text.replace('ى', 'ي')
    # Remove non-Arabic characters except spaces and numbers
    text = re.sub(r'[^\u0600-\u06FF\s0-9]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class DocumentContent:
    def __init__(self):
        self.sections = {}  # Changed from defaultdict to regular dict
        self.section_text = {}  # Changed from defaultdict to regular dict
        self.current_section = None
        self.vectorizer = None
        self.vectors = None
        self.content = []

def is_heading(paragraph):
    """Improved Arabic heading detection"""
    text = paragraph.text.strip()
    if not text:
        return False
        
    # Define Arabic heading keywords
    heading_keywords = [
        'كلمة', 'مجلس', 'تعريف', 'ملخص', 'إنجازات', 'مقدمة', 'برنامج',
        'الأهداف', 'النتائج', 'التوصيات', 'الخطة', 'المشاريع', 'الرؤية',
        'الرسالة', 'القيم', 'التحديات', 'المبادرات'
    ]
    
    # Check for heading style
    if paragraph.style and 'heading' in str(paragraph.style.name).lower():
        return True
    
    # Check if runs are bold
    if paragraph.runs and all(run.bold for run in paragraph.runs):
        if len(text) > 150:  # Too long to be a heading
            return False
            
        # Check for heading patterns
        if (any(text.startswith(kw) for kw in heading_keywords) or
            any(text.endswith(marker) for marker in [':', '؛', '-', '.']) or
            len(text.split()) <= 7):
            return True
            
    return False

def process_text_chunk(text, max_length=1500):
    """Process text into smaller chunks with better Arabic sentence handling"""
    if not text or len(text) <= max_length:
        return [text] if text else []

    # Split on Arabic sentence endings
    sentences = re.split(r'[.!؟]', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def load_docx_content():
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        files = os.listdir(current_dir)
        docx_file = next((f for f in files if f.strip().endswith('arabic_file.docx')), None)
        
        if not docx_file:
            logger.error("Document not found")
            return None
            
        doc_path = os.path.join(current_dir, docx_file)
        logger.info(f"Loading document from: {doc_path}")
        
        doc = Document(doc_path)
        doc_content = DocumentContent()
        
        # Process document content
        current_text = ""
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            if is_heading(paragraph):
                if current_text and doc_content.current_section:
                    processed_text = preprocess_arabic_text(current_text)
                    doc_content.section_text[doc_content.current_section] = processed_text
                    doc_content.sections[doc_content.current_section] = current_text
                    
                    # Add to content list for vectorization
                    doc_content.content.append({
                        'text': current_text,
                        'section': doc_content.current_section
                    })
                
                doc_content.current_section = text
                current_text = ""
                continue
            
            if doc_content.current_section:
                current_text += " " + text
        
        # Process final section
        if current_text and doc_content.current_section:
            processed_text = preprocess_arabic_text(current_text)
            doc_content.section_text[doc_content.current_section] = processed_text
            doc_content.sections[doc_content.current_section] = current_text
            
            doc_content.content.append({
                'text': current_text,
                'section': doc_content.current_section
            })
        
        # Log found sections
        logger.info(f"Found {len(doc_content.sections)} sections:")
        for section in doc_content.sections.keys():
            logger.info(f"- {section}")
        
        # Initialize vectorizer with Arabic-specific settings
        doc_content.vectorizer = TfidfVectorizer(
            max_features=2000,  # Reduced from 5000
            ngram_range=(1, 2),  # Reduced from (1,3)
            preprocessor=preprocess_arabic_text,
            token_pattern=r'[\u0600-\u06FF\s]+',
        )
        
        # Prepare texts for vectorization
        texts = []
        for item in doc_content.content:
            # Include section name for context but without doubling
            section_text = f"{item['section']} {item['text']}"
            texts.append(preprocess_arabic_text(section_text))
            
        doc_content.vectors = doc_content.vectorizer.fit_transform(texts)
        
        # Clean up
        gc.collect()
        
        logger.info(f"Document processed successfully with {len(doc_content.content)} sections")
        return doc_content
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return None

DOC_PROCESSOR = load_docx_content()

def find_relevant_content(question):
    """Find relevant content using improved similarity search"""
    if not DOC_PROCESSOR:
        return []
    
    processed_question = preprocess_arabic_text(question)
    logger.debug(f"Processed question: {processed_question}")
    
    # Get similarity scores
    question_vector = DOC_PROCESSOR.vectorizer.transform([processed_question])
    similarities = np.array(DOC_PROCESSOR.vectors.dot(question_vector.T).toarray()).flatten()
    
    # Get all matches above minimum threshold
    min_similarity = 0.01
    matching_indices = np.where(similarities > min_similarity)[0]
    
    # Group by sections and calculate scores
    section_scores = defaultdict(float)
    section_content = {}
    
    for idx in matching_indices:
        score = float(similarities[idx])
        content = DOC_PROCESSOR.content[idx]
        section = content['section']
        
        # Keep highest scoring content for each section
        if section not in section_content or score > section_scores[section]:
            section_content[section] = content['text']
            section_scores[section] = score
    
    # Get all relevant sections
    relevant_content = []
    for section, score in sorted(section_scores.items(), key=lambda x: x[1], reverse=True):
        if score > min_similarity:
            relevant_content.append({
                'section': section,
                'text': section_content[section],
                'score': score
            })
    
    logger.debug(f"Found {len(relevant_content)} relevant sections")
    return relevant_content

def ask_claude(question, context):
    """Send the document and question to Claude API."""
    token_estimate = len(context + question) // 4
    
    if token_estimate > 45000:
        logger.warning("Context too long, truncating...")
        return "النص طويل جداً. يرجى تقسيم سؤالك إلى أجزاء أصغر."
        
    messages = [
        {
            "role": "user",
            "content": f"""هنا نص التقرير. أجب على سؤال المستخدم بناءً على المعلومات الواردة في النص فقط. إذا كانت المعلومة موجودة، اذكر القسم الذي وجدتها فيه. إذا لم تكن المعلومة موجودة، وضح ذلك بشكل صريح.

النص:
{context}

سؤال المستخدم: {question}

تعليمات مهمة:
1. إذا وجدت المعلومة في النص، اذكر القسم بالتحديد
2. انقل النص الأصلي الذي يحتوي على الإجابة
3. إذا لم تجد المعلومة، قل ذلك بوضوح
4. لا تستنتج أو تخمن - اعتمد فقط على ما ورد في النص"""
        }
    ]
    
    headers = {
        "anthropic-version": "2023-06-01",
        "x-api-key": CLAUDE_API_KEY,
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "temperature": 0.1,
        "messages": messages
    }

    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        elif response.status_code == 429:
            return "تم تجاوز حد الطلبات. يرجى المحاولة مرة أخرى لاحقاً."
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return f"حدث خطأ في معالجة الطلب."
            
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return "حدث خطأ في معالجة الطلب."

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "لم يتم تقديم سؤال"}), 400

    logger.info(f"Received question: {question}")
    
    if not DOC_PROCESSOR:
        return jsonify({"error": "لم يتم تحميل الوثيقة بشكل صحيح."}), 500

    relevant_content = find_relevant_content(question)
    logger.debug(f"Found {len(relevant_content)} relevant sections")
    
    if not relevant_content:
        return jsonify({"answer": "عذرًا، لا توجد معلومات ذات صلة في التقرير."})

    # Format context with section boundaries and scores
    context_parts = []
    for item in relevant_content:
        context_parts.append(f"""
=== {item['section']} (درجة التطابق: {item['score']:.2f}) ===
{item['text']}
=== نهاية {item['section']} ===
""")
    
    context = "\n\n".join(context_parts)
    logger.debug(f"Context length: {len(context)} characters")
    
    answer = ask_claude(question, context)
    return jsonify({"answer": answer})

@app.route('/api/sections', methods=['GET'])
def list_sections():
    """Debug endpoint to list all document sections"""
    if not DOC_PROCESSOR:
        return jsonify({"error": "Document not loaded"}), 500
        
    sections = []
    for section, content in DOC_PROCESSOR.sections.items():
        sections.append({
            "title": section,
            "char_count": len(content),
        })
    
    return jsonify({"sections": sections})

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "document_loaded": bool(DOC_PROCESSOR),
        "document_path": DOCUMENT_PATH,
        "sections_count": len(DOC_PROCESSOR.sections) if DOC_PROCESSOR else 0
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
