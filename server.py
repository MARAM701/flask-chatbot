from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
from docx import Document
import re
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

DOCUMENT_PATH = os.getenv('DOCUMENT_PATH', 'arabic_file.docx')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# Define known headers in order of appearance
KNOWN_HEADERS = [
    "كلمة معالي رئيس مجلس الإدارة",
    "كلمة معالي رئيس المدينة",
    "مجلس إدارة مدينة الملك عبدالعزيز للعلوم والتقنية",
    "تعريف المصطلحات والاختصارات",
    "الملخص التنفيذي",
    "إنجازات العام في أرقام"
]

# Add TOC page mapping
TOC_PAGE_MAP = {
    "كلمة معالي رئيس مجلس الإدارة": 8,
    "كلمة معالي رئيس المدينة": 10,
    "مجلس إدارة مدينة الملك عبدالعزيز للعلوم والتقنية": 12,
    "تعريف المصطلحات والاختصارات": 14,
    "الملخص التنفيذي": 18,
    "إنجازات العام في أرقام": 20
}

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
            
            # Initialize with first header or default
            current_section = KNOWN_HEADERS[0] if KNOWN_HEADERS else "مقدمة"
            current_content = []
            
            # Process document
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                
                # Check if this paragraph matches any known header
                if text in KNOWN_HEADERS:
                    # Save previous section content if exists
                    if current_content:
                        self.sections[current_section] = '\n'.join(current_content)
                    # Start new section
                    current_section = text
                    current_content = []
                    logger.debug(f"Found header: {text}")
                else:
                    # This is normal text ("عادي"), add to current section
                    current_content.append(text)
            
            # Save last section
            if current_content:
                self.sections[current_section] = '\n'.join(current_content)
            
            # Store full document text
            self.document_text = '\n\n'.join(para.text for para in doc.paragraphs if para.text.strip())
            
            # Verify all expected sections were found
            found_headers = set(self.sections.keys())
            expected_headers = set(KNOWN_HEADERS)
            missing_headers = expected_headers - found_headers
            
            if missing_headers:
                logger.warning(f"Missing expected headers: {missing_headers}")
            
            logger.info(f"Document loaded successfully with {len(self.sections)} sections")
            logger.info(f"Found sections: {list(self.sections.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}", exc_info=True)
            return False

def process_gpt_response(gpt_response):
    """Add page number to GPT's response"""
    section_match = re.search(r"القسم: (.+?)(?:\n|$)", gpt_response, re.MULTILINE)
    if not section_match:
        return gpt_response
    
    section_name = section_match.group(1).strip()
    page_number = TOC_PAGE_MAP.get(section_name, "غير متوفر")
    
    # Add page number to the end of the response
    return f"{gpt_response}\nالصفحة: {page_number}"

def ask_gpt4(question, context):
    """Send the document and question to OpenAI GPT-4 API."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = """أنت مساعد متخصص في تحليل النصوص العربية والإجابة على الأسئلة بدقة عالية.
    يجب عليك الالتزام بالقواعد التالية بشكل صارم:

    1. إذا وجدت المعلومة في النص، اذكر القسم بالتحديد.
    2. انقل النص الأصلي الذي يحتوي على الإجابة.
    3. إذا لم تجد المعلومة، قل ذلك بوضوح.
    4. لا تستنتج أو تخمن - اعتمد فقط على ما ورد في النص.
    5. إذا كانت المعلومة قائمة بالأسماء أو في شكل قائمة، قم بذكر القائمة كما هي.
    6. تعامل مع القوائم والنقاط كجزء من المعلومات في النص.
    7. لا تتجاهل الأسطر القصيرة التي قد تكون ذات مغزى.

    نموذج الإجابة: 
    - الإجابة: [إجابتك المبنية على النص فقط] 
    - النص الأصلي: [النص الحرفي من المستند]
    - القسم: [اسم القسم الذي وجدت فيه المعلومة]
    
    إذا لم تجد المعلومة:
    - لم أجد معلومات في النص تجيب على هذا السؤال."""

    user_message = f"""هنا نص التقرير. أجب على سؤال المستخدم بناءً على المعلومات الواردة في النص فقط.

النص:
{context}

سؤال المستخدم: {question}

تذكر:
- اذكر القسم الذي وجدت فيه المعلومة
- انقل النص الأصلي حرفياً
- لا تستنتج أو تخمن"""

    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        gpt_response = response.choices[0].message.content
        return process_gpt_response(gpt_response)
            
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
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
    
    answer = ask_gpt4(question, context)
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
            "page": TOC_PAGE_MAP.get(section, "غير متوفر")
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
