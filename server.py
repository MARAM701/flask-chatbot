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
        self.sections = defaultdict(list)
        self.current_section = None
        self.current_page = 1
        self.content = []
        self.vectorizer = None
        self.vectors = None
        self.section_text = defaultdict(str)
        self.raw_text = defaultdict(str)

def is_heading(paragraph):
    # Check for heading styles including Arabic
    heading_keywords = ['كلمة', 'مجلس الإدارة', 'تعريف', 'ملخص', 'إنجازات', 'مقدمة', 'برنامج']
    if paragraph.style and any(style in paragraph.style.name.lower() for style in 
        ['heading', 'title', 'header', 'العنوان', 'عنوان', 'رئيسي', 'فرعي']):
        return True
    
    if paragraph.runs and paragraph.runs[0].bold:
        text = paragraph.text.strip()
        if len(text) < 100 and (
            any(text.endswith(marker) for marker in [':', '：', '：', '：', '-', '.']) or
            any(keyword in text for keyword in heading_keywords)
        ):
            return True
        return True
    return False

def process_text_chunk(text, max_length=3000):
    """Split text into chunks with overlap for better context"""
    if not text or len(text) <= max_length:
        return [text] if text else []

    sentences = re.split(r'(?<=\.)|(?<=؟)|(?<=!)', text)
    chunks = []
    current_chunk = []
    current_length = 0
    overlap = 500  # Increase overlap for better context

    for sentence in sentences:
        sentence = sentence.strip()
        if current_length + len(sentence) > max_length - overlap and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            # Keep last few sentences for overlap
            overlap_text = ' '.join(current_chunk[-3:])
            current_chunk = [overlap_text]
            current_length = len(overlap_text)
        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_token_estimate(text):
    """Rough token estimate - 4 chars per token"""
    return len(text) // 4

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
                    chunks = process_text_chunk(current_text)
                    for chunk in chunks:
                        doc_content.sections[doc_content.current_section].append({
                            'text': chunk,
                            'page': doc_content.current_page
                        })
                    doc_content.section_text[doc_content.current_section] = preprocess_arabic_text(current_text)
                    doc_content.raw_text[doc_content.current_section] = current_text
                
                doc_content.current_section = text
                current_text = ""
                continue
            
            if doc_content.current_section:
                current_text += " " + text
        
        # Process final section
        if current_text and doc_content.current_section:
            chunks = process_text_chunk(current_text)
            for chunk in chunks:
                doc_content.sections[doc_content.current_section].append({
                    'text': chunk,
                    'page': doc_content.current_page
                })
            doc_content.section_text[doc_content.current_section] = preprocess_arabic_text(current_text)
            doc_content.raw_text[doc_content.current_section] = current_text
        
        # Prepare content for vectorization
        for section, chunks in doc_content.sections.items():
            for chunk in chunks:
                doc_content.content.append({
                    'text': chunk['text'],
                    'section': section,
                    'page': chunk['page']
                })
        
        # Initialize TF-IDF with section-aware text representation
        doc_content.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            preprocessor=preprocess_arabic_text,
            token_pattern=r'[\u0600-\u06FF\s]+',
        )
        # Include section headers in vectorization for better context
        texts = []
        for item in doc_content.content:
            section_text = f"{item['section']} {item['section']} {item['text']}"  # Double section weight
            texts.append(preprocess_arabic_text(section_text))
            
        doc_content.vectors = doc_content.vectorizer.fit_transform(texts)
        
        logger.info(f"Document processed successfully with {len(doc_content.content)} chunks")
        return doc_content
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return None

DOC_PROCESSOR = load_docx_content()

def find_relevant_content(question, top_k=50):
    """Find relevant content using enhanced TF-IDF similarity search"""
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
    section_groups = defaultdict(list)
    section_scores = defaultdict(float)
    
    for idx in matching_indices:
        score = float(similarities[idx])
        content = DOC_PROCESSOR.content[idx].copy()
        content['score'] = score
        
        section = content['section']
        section_groups[section].append(content)
        section_scores[section] += score
    
    # Sort sections by total relevance
    ranked_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug("Ranked sections with scores:")
    for section, score in ranked_sections[:10]:
        logger.debug(f"Section: {section}, Score: {score}")
    
    # Get best content from top sections
    relevant_content = []
    seen_sections = set()
    
    # First, add highest scoring content from each section
    for section, total_score in ranked_sections:
        if total_score > min_similarity and section not in seen_sections:
            # Get best chunk from this section
            best_chunk = max(section_groups[section], key=lambda x: x['score'])
            best_chunk['text'] = DOC_PROCESSOR.raw_text[section]
            relevant_content.append(best_chunk)
            seen_sections.add(section)
            
            if len(relevant_content) >= min(10, len(DOC_PROCESSOR.sections)):
                break
    
    logger.debug(f"Found content in {len(relevant_content)} sections")
    return relevant_content

def ask_claude(question, context):
    """Send the document and question to Claude API."""
    token_estimate = get_token_estimate(context + question)
    
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
    for item in sorted(relevant_content, key=lambda x: x['score'], reverse=True):
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
            "char_count": len(DOC_PROCESSOR.section_text[section]),
            "chunk_count": len(content)
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
