from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
import re
import json
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

JSON_FILE_PATH = os.getenv('JSON_FILE_PATH', 'report_2016.json')
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

# Load JSON data at server startup
REPORT_DATA = []

def load_json_data():
    """Load and preprocess the JSON file at server startup"""
    global REPORT_DATA
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        json_path = os.path.join(current_dir, JSON_FILE_PATH)
        logger.info(f"Loading JSON file from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as file:
            REPORT_DATA = json.load(file)
            
        logger.info(f"Successfully loaded {len(REPORT_DATA)} sections from JSON file")
        return True
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}", exc_info=True)
        return False

def ask_gpt4(question, context):
    """Send the document and question to OpenAI GPT-4 API."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = """أنت مساعد متخصص في تحليل النصوص العربية والإجابة على الأسئلة بدقة عالية.
    يجب عليك البحث في جميع الأقسام المتوفرة والالتزام بالقواعد التالية بشكل صارم:

    1. إذا كانت المعلومات مأخوذة من قسم واحد فقط:
    **الإجابة:** 
    [إجابتك المبنية على النص] [1]

    **النص الأصلي:**
    "[أول 50 حرف من النص المقتبس]..."
    
    📖 المصدر:
    [اسم الملف] - [اسم القسم]

    2. إذا كانت المعلومات مأخوذة من عدة أقسام:
    **الإجابة:**
    [إجابتك المبنية على النص مع رقم المرجع بعد كل معلومة]

    **النص الأصلي:**
    [1]: "[أول 30 حرف من النص المقتبس]..."
    [2]: "[أول 30 حرف من النص المقتبس]..."
    
    📖 المصادر:
    [1]: [اسم الملف] - [اسم القسم]
    [2]: [اسم الملف] - [اسم القسم]

    3. إذا لم تجد المعلومة في النص، اكتب:
    **الإجابة:** عذراً، لم أجد معلومات في النص تجيب على هذا السؤال.

    4. التزم بالقواعد التالية:
    - اعتمد فقط على المعلومات الموجودة في النص
    - ابحث في جميع الأقسام قبل تقديم الإجابة
    - أضف رقم المرجع [N] بعد كل معلومة مقتبسة
    - اقتبس فقط أول 50 حرف من النص الأصلي متبوعة بثلاث نقاط (...)
    - رتب المراجع حسب ظهورها في الإجابة"""

    user_message = f"""قم بالبحث في جميع أقسام النص التالي وأجب على سؤال المستخدم بناءً على المعلومات الواردة.
    تأكد من ذكر جميع المصادر ذات الصلة.

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
            max_tokens=1500
        )
        
        gpt_response = response.choices[0].message.content
        return process_gpt_response(gpt_response)
            
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return "حدث خطأ في معالجة الطلب."

def process_gpt_response(gpt_response):
    """Format GPT response with simplified reference format"""
    # Check if it's a "no information found" response
    if "عذراً، لم أجد معلومات" in gpt_response:
        return gpt_response
        
    # Handle both single and multiple sources
    sources_section = None
    
    if "📖 المصدر:" in gpt_response:
        # Convert single source format to multiple source format
        gpt_response = gpt_response.replace("📖 المصدر:", "📖 المصادر:\n[1]:")
        sources_section = re.search(r'📖 المصادر:(.*?)(?=\*\*|\n\n|\Z)', gpt_response, re.DOTALL)
    elif "📖 المصادر:" in gpt_response:
        sources_section = re.search(r'📖 المصادر:(.*?)(?=\*\*|\n\n|\Z)', gpt_response, re.DOTALL)
    
    # We're not modifying the sources as we don't need page numbers anymore
    # The GPT response already contains the correct format for the sources
    
    return gpt_response

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "لم يتم تقديم سؤال"}), 400

    logger.info(f"Received question: {question}")
    
    if not REPORT_DATA:
        return jsonify({"error": "لم يتم تحميل البيانات بشكل صحيح."}), 500

    # Format JSON data as context
    context_parts = []
    for section in REPORT_DATA:
        file_name = section.get('file_name', '')
        section_header = section.get('section_header', '')
        content = section.get('content', '')
        
        context_parts.append(f"""
=== {section_header} ===
{content}
=== نهاية {section_header} ===
""")
    
    context = "\n\n".join(context_parts)
    
    answer = ask_gpt4(question, context)
    return jsonify({"answer": answer})

@app.route('/api/sections', methods=['GET'])
def list_sections():
    """Endpoint to list all document sections"""
    if not REPORT_DATA:
        return jsonify({"error": "JSON data not loaded"}), 500
        
    sections = []
    for section in REPORT_DATA:
        sections.append({
            "file_name": section.get('file_name', ''),
            "title": section.get('section_header', ''),
            "char_count": len(section.get('content', '')),
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
        "document_loaded": bool(REPORT_DATA),
        "json_file_path": JSON_FILE_PATH,
        "sections_count": len(REPORT_DATA)
    }), 200

# Load the JSON data when the server starts
load_json_data()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
