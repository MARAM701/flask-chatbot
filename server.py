from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
from docx import Document
import re
import requests

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

class DocumentProcessor:
    def __init__(self):
        self.sections = {}
        self.document_text = ""

    def load_document(self):
        try:
            current_dir = os.getcwd()
            logger.info(f"Current working directory: {current_dir}")
            
            files = os.listdir(current_dir)
            docx_file = next((f for f in files if f.strip().endswith('arabic_file.docx')), None)
            
            if not docx_file:
                logger.error("Document not found")
                return False
                
            doc_path = os.path.join(current_dir, docx_file)
            logger.info(f"Loading document from: {doc_path}")
            
            doc = Document(doc_path)
            current_section = "مقدمة"
            current_content = []
            
            # Process document and identify sections
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                
                # Simple header detection - bold text
                if paragraph.runs and paragraph.runs[0].bold:
                    if current_content:
                        self.sections[current_section] = '\n'.join(current_content)
                    current_section = text
                    current_content = []
                else:
                    current_content.append(text)
            
            # Save last section
            if current_content:
                self.sections[current_section] = '\n'.join(current_content)
            
            # Store full document text
            self.document_text = '\n\n'.join(para.text for para in doc.paragraphs if para.text.strip())
            
            logger.info(f"Document loaded successfully with {len(self.sections)} sections")
            return True
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}", exc_info=True)
            return False

def ask_claude(question, context):
    """Send the document and question to Claude API."""
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
        "model": "claude-3-5-haiku-20241022",
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

# Initialize document processor
DOC_PROCESSOR = DocumentProcessor()
DOC_PROCESSOR.load_document()

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "لم يتم تقديم سؤال"}), 400

    logger.info(f"Received question: {question}")
    
    if not DOC_PROCESSOR.sections:
        return jsonify({"error": "لم يتم تحميل الوثيقة بشكل صحيح."}), 500

    # Format document sections
    context_parts = []
    for section, content in DOC_PROCESSOR.sections.items():
        context_parts.append(f"""
=== {section} ===
{content}
=== نهاية {section} ===
""")
    
    context = "\n\n".join(context_parts)
    
    answer = ask_claude(question, context)
    return jsonify({"answer": answer})

@app.route('/api/sections', methods=['GET'])
def list_sections():
    """Debug endpoint to list all document sections"""
    if not DOC_PROCESSOR.sections:
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
        "document_loaded": bool(DOC_PROCESSOR.sections),
        "document_path": DOCUMENT_PATH,
        "sections_count": len(DOC_PROCESSOR.sections)
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
