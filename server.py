from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
from docx import Document
import re
import google.generativeai as genai  # Changed from OpenAI to Gemini

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

DOCUMENT_PATH = os.getenv('DOCUMENT_PATH', 'test_second.docx')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # New API key for Gemini

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
        self.file_header = "" 

    def load_document(self):
        try:
            current_dir = os.getcwd()
            logger.info(f"Current working directory: {current_dir}")

            doc_path = os.path.join(current_dir, DOCUMENT_PATH)
            if not os.path.exists(doc_path):
                logger.error("Document not found")
                return False
            logger.info(f"Loading document from: {doc_path}")

            doc = Document(doc_path) 
                       # Initialize with first header or default
            current_section = None
            current_content = []
            header_pattern = re.compile(r'^(#{3,})\s*(.+?)\s*\1$')




            # Process document using regex to detect headers
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue

                match = header_pattern.match(text)
                if match:
                    level = len(match.group(1))
                    header_text = match.group(2).strip()
                    logger.debug(f"Found header: {header_text} with level {level}")
                    if level == 3:
                        # إذا كانت العلامات ثلاث (###): هذا عنوان الملف
                        self.file_header = text
                        continue  # لا تُضيف إلى أي قسم
                    elif level >= 4:
                        # إذا كانت العلامات أربع أو أكثر: هذا عنوان قسم
                        if current_section and current_content:
                            self.sections[current_section] = '\n'.join(current_content)
                        current_section = header_text
                        current_content = []
                        continue
                else:
                    # إذا لم يتطابق مع النمط، يُضاف إلى المحتوى الحالي
                    current_content.append(text) 
 
                        # Save last section if exists
            if current_section and current_content:
                self.sections[current_section] = '\n'.join(current_content)
            
            # Build full document text with file header and sections
            parts = []
            if self.file_header:
                parts.append(self.file_header)
            for section, content in self.sections.items():
                parts.append(f"=== {section} ===\n{content}\n=== نهاية {section} ===")
            self.document_text = "\n\n".join(parts)


            return True

        except Exception as e:
            logger.error(f"Error loading document: {str(e)}", exc_info=True)
            return False


def ask_gemini(question, context):
    """Send the document and question to Gemini API."""
    genai.configure(api_key=GEMINI_API_KEY)  # Configure the library with your API key
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

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
        # Combine system_prompt and user_message into a single message
        combined_message = system_prompt + "\n\n" + user_message

        response = model.generate_content(
            combined_message,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1500
            }
        )
        gemini_response = response.text

        return process_gpt_response(gemini_response)

    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return "حدث خطأ في معالجة الطلب."

def process_gpt_response(gpt_response):
    """Format GPT response with numbered references using detected file name and section header."""
    # Check if the response indicates no information was found
    if "عذراً، لم أجد معلومات" in gpt_response:
        return gpt_response

    # Extract the sources section from the GPT response
    sources_section = None
    if "📖 المصدر:" in gpt_response:
        # Convert single source format to multiple sources format
        gpt_response = gpt_response.replace("📖 المصدر:", "📖 المصادر:\n[1]:")
        sources_section = re.search(r'📖 المصادر:(.*?)(?=\*\*|\n\n|\Z)', gpt_response, re.DOTALL)
    elif "📖 المصادر:" in gpt_response:
        sources_section = re.search(r'📖 المصادر:(.*?)(?=\*\*|\n\n|\Z)', gpt_response, re.DOTALL)

    # Dynamically extract the file name from the global DocumentProcessor instance's file_header
    file_name = "Unknown File"
    try:
        # Assume DOC_PROCESSOR is a global instance of DocumentProcessor
        header_text = DOC_PROCESSOR.file_header  # e.g., "### اسم الملف: التقرير السنوي ٢٠٢٢ ###"
        match = re.search(r'اسم الملف:\s*(.*?)\s*#', header_text)
        if match:
            file_name = match.group(1).strip()
    except Exception as e:
        file_name = "Unknown File"

    # If sources section is found, process each reference line
    if sources_section:
        sources_text = sources_section.group(1)
        modified_sources = []
        # Iterate over each reference line using regex
        for ref_match in re.finditer(r'\[(\d+)\]:\s*(.*?)(?=\n|$)', sources_text):
            ref_num = ref_match.group(1)
            section_name = ref_match.group(2).strip()
            # Format reference as: [ref_num]: {file_name} - {section_name}
            modified_sources.append(f'[{ref_num}]: {file_name} - {section_name}')
        if modified_sources:
            new_sources = '📖 المصادر:\n' + '\n'.join(modified_sources)
            gpt_response = re.sub(
                r'📖 المصادر:.*?(?=\*\*|\n\n|\Z)',
                new_sources,
                gpt_response,
                flags=re.DOTALL
            )

    return gpt_response





# Create a global instance of DocumentProcessor and load the document
DOC_PROCESSOR = DocumentProcessor()
if not DOC_PROCESSOR.load_document():
    logger.error("Failed to load the document.")


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

    answer = ask_gemini(question, context)
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
