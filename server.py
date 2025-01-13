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
    "انجازات العام في أرقام": 20
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}", exc_info=True)
            return False

def process_gpt_response(gpt_response):
    """Format GPT response maintaining emoji source reference and adding numbered references"""
    # First find all section references
    section_matches = re.finditer(r"المصدر: ([^-\n]+)", gpt_response)
    
    # Collect all unique sections while preserving the response format
    sections = []
    seen_sections = set()
    
    result = gpt_response
    
    # Process each section reference
    for match in section_matches:
        section_name = match.group(1).strip()
        if section_name not in seen_sections:
            sections.append(section_name)
            seen_sections.add(section_name)
            
            # Get page number for this section
            page_number = TOC_PAGE_MAP.get(section_name, "غير متوفر")
            
            # Replace the source reference with emoji format, handling potential duplicate emojis
            result = re.sub(
                f'📖* *المصدر: {re.escape(section_name)}',
                f'📖 المصدر: {section_name} - صفحة {page_number}',
                result
            )
            # Clean up any remaining duplicate emojis
            result = re.sub(r'📖\s*📖\s*', '📖 ', result)
    
    # Add numbered references at the end if there are sections
    if sections:
        result += "\n\nالمراجع:\n"
        for idx, section in enumerate(sections, 1):
            page_number = TOC_PAGE_MAP.get(section, "غير متوفر")
            result += f"[{idx}] {section} - صفحة {page_number}\n"
    
    return result

def ask_gpt4(question, context):
    """Send the document and question to OpenAI GPT-4 API."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = """أنت مساعد متخصص في تحليل النصوص العربية والإجابة على الأسئلة بدقة عالية.
    يجب عليك الالتزام بالقواعد التالية بشكل صارم:

    1. قدم الإجابة بالترتيب التالي:
    أولاً: الإجابة الكاملة
    **الإجابة:** [اكتب إجابة كاملة ومنظمة تجمع المعلومات من كل المصادر]

    ثانياً: النصوص المصدرية لكل قسم:
    📖 المصدر: [اسم القسم الأول] - صفحة [رقم]
    **النص الأصلي:** "[النص الحرفي من القسم الأول]"

    📖 المصدر: [اسم القسم الثاني] - صفحة [رقم]
    **النص الأصلي:** "[النص الحرفي من القسم الثاني]"
    (وهكذا لكل قسم)

    ثالثاً: قائمة المراجع في النهاية

    2. عند الإجابة على أسئلة الإنجازات أو النتائج:
    - نظم الإجابة في نقاط مرقمة أو مرتبة
    - اجمع المعلومات المتشابهة تحت عناوين موحدة
    - رتب النقاط حسب الأهمية أو التسلسل المنطقي

    3. التزم بالقواعد التالية:
    - اعتمد فقط على المعلومات الموجودة في النص
    - لا تستنتج أو تخمن
    - انقل النص الأصلي حرفياً لكل قسم
    - حافظ على ترتيب: الإجابة، ثم النصوص المصدرية، ثم المراجع"""

    1. قدم الإجابة بالتنسيق التالي بالضبط:
    **الإجابة:** [إجابتك المبنية على النص]
    
    إذا كانت المعلومات من قسم واحد:
    📖 المصدر: [اسم القسم]
    **النص الأصلي:** "[النص الحرفي من المستند]"
    
    إذا كانت المعلومات من عدة أقسام:
    📖 المصدر: [اسم القسم الأول]
    **النص الأول:** "[النص الحرفي الأول]"
    📖 المصدر: [اسم القسم الثاني]
    **النص الثاني:** "[النص الحرفي الثاني]"
    (وهكذا لكل قسم)

    2. إذا لم تجد المعلومة في النص، اكتب:
    **الإجابة:** عذراً، لم أجد معلومات في النص تجيب على هذا السؤال.

    3. التزم بالقواعد التالية:
    - اعتمد فقط على المعلومات الموجودة في النص
    - لا تستنتج أو تخمن
    - انقل النص الأصلي حرفياً لكل قسم
    - اذكر المصدر لكل قسم بالتنسيق المطلوب
    - تعامل مع القوائم والنقاط كجزء من المعلومات"""

    user_message = f"""هنا نص التقرير. أجب على سؤال المستخدم بناءً على المعلومات الواردة في النص فقط.

النص:
{context}

سؤال المستخدم: {question}"""

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
