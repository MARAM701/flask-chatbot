from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
from docx import Document
import re
import requests

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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('server')

# Claude API configuration
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

class DocumentContent:
    def __init__(self):
        self.sections = {}
        self.current_section = None
        self.content = []

def is_heading(paragraph):
    """Check if a paragraph is a heading based on style and formatting"""
    # Check for heading styles including Arabic
    if paragraph.style and any(style in paragraph.style.name.lower() for style in 
        ['heading', 'title', 'header', 'العنوان', 'عنوان', 'رئيسي', 'فرعي']):
        return True
    
    # Check for bold text
    if paragraph.runs and paragraph.runs[0].bold:
        # Check if it looks like a heading (short, ends with common markers)
        text = paragraph.text.strip()
        if len(text) < 100 and any(text.endswith(marker) for marker in [':', '：', '：', '：', '-', '.']):
            return True
        return True
        
    return False

def load_docx_content():
    try:
        doc = Document('arabic_file.docx')
        doc_content = DocumentContent()
        
        logger.info("Starting document processing")
        current_section = "مقدمة"  # Default section

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            logger.info(f"Processing: {text[:50]}... | Style: {paragraph.style.name if paragraph.style else 'No style'}")
            
            if is_heading(paragraph):
                current_section = text
                logger.info(f"Found header: {text}")
                continue
            
            # Add content with section
            doc_content.content.append({
                'text': text,
                'section': current_section
            })
        
        logger.info(f"Successfully loaded document content with {len(doc_content.content)} paragraphs")
        return doc_content.content
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        return []

# Load report content when server starts
DOCUMENT_CONTENT = load_docx_content()

def find_relevant_content(question):
    """Find relevant paragraphs based on the question using improved matching"""
    relevant_content = []
    question_words = set(re.findall(r'[\u0600-\u06FF]+', question.lower()))  # Arabic words only
    
    for content in DOCUMENT_CONTENT:
        content_words = set(re.findall(r'[\u0600-\u06FF]+', content['text'].lower()))
        # Calculate word overlap
        overlap = len(question_words.intersection(content_words))
        if overlap > 0:
            content['relevance_score'] = overlap
            relevant_content.append(content)
    
    # Sort by relevance and take top results
    relevant_content.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_content[:5]  # Return top 5 most relevant sections

@app.route('/')
def home():
    return "Server is running"

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "لم يتم تقديم سؤال"}), 400
            
        logger.info(f"Received question: {question}")
        
        # Find relevant content
        relevant_content = find_relevant_content(question)
        
        # If no relevant content found
        if not relevant_content:
            return jsonify({
                "answer": "عذرًا، لا توجد معلومات ذات صلة في التقرير."
            })

        # Format content for Claude with sources
        context = "\n\n".join([
            f"{item['text']}\n📖 المصدر: {item['section']}"
            for item in relevant_content
        ])

        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية. استخدم المعلومات التالية للإجابة على الأسئلة:

النص:
{context}

قواعد مهمة:
1. اعتمد فقط على المعلومات الموجودة في النص المقدم.
2. إذا كانت المعلومة غير موجودة في النص، قل ذلك بوضوح.
3. اذكر القسم الذي وجدت فيه المعلومة.
4. انقل النص الدقيق الذي يحتوي على الإجابة.
5. لا تستنتج أو تضيف معلومات غير موجودة في النص.
6. قدم إجابة مباشرة ومختصرة.

السؤال: {question}

يرجى الإجابة مع ذكر المصدر بدقة."""
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

            response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                answer = response.json()["content"][0]["text"]
                return jsonify({"answer": answer})
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return jsonify({
                    "error": "عذراً، حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى."
                }), 500
                
        except Exception as api_error:
            logger.error(f"API error: {str(api_error)}")
            return jsonify({
                "error": "عذراً، حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى."
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "عذراً، حدث خطأ في معالجة طلبك. الرجاء المحاولة مرة أخرى."
        }), 500

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
        "document_loaded": bool(DOCUMENT_CONTENT),
        "sections_count": len(set(item['section'] for item in DOCUMENT_CONTENT))
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
