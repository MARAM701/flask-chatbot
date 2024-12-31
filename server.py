from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
from docx import Document
import re

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

# Create OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class DocumentContent:
    def __init__(self):
        self.sections = {}
        self.current_section = "مقدمة"
        self.current_page = 1
        self.content = []

def load_docx_content():
    try:
        doc = Document('arabic_file.docx')
        doc_content = DocumentContent()
        
        # Regular expression for page markers
        page_marker_pattern = re.compile(r'Page\s+(\d+)')
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            # Check for page markers
            page_match = page_marker_pattern.search(text)
            if page_match:
                doc_content.current_page = int(page_match.group(1))
                continue
            
            # Check if it's a heading (you might need to adjust this based on your document structure)
            if paragraph.style.name.startswith('Heading'):
                doc_content.current_section = text
                continue
            
            # Store the content with metadata
            doc_content.content.append({
                'text': text,
                'section': doc_content.current_section,
                'page': doc_content.current_page
            })
        
        logger.info("Successfully loaded document content with sections and pages")
        return doc_content.content
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        return []

# Load report content when server starts
DOCUMENT_CONTENT = load_docx_content()

def find_relevant_content(question):
    """Find relevant paragraphs based on the question"""
    relevant_content = []
    question_words = set(question.lower().split())
    
    for content in DOCUMENT_CONTENT:
        # Simple keyword matching (you can enhance this)
        content_words = set(content['text'].lower().split())
        if any(word in content_words for word in question_words):
            relevant_content.append(content)
    
    return relevant_content

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

        # Format content for AI with sources
        context = "\n\n".join([
            f"{item['text']}\n📖 المصدر: {item['section']} - صفحة {item['page']}"
            for item in relevant_content
        ])

        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية لعام 2023. استخدم المعلومات التالية للإجابة على الأسئلة:

                        {context}

                        قواعد مهمة:
                        1. اعتمد فقط على المعلومات الموجودة في النص أعلاه.
                        2. أجب باللغة العربية الفصحى.
                        3. قدم إجابة مختصرة ومنظمة مع ذكر المصدر لكل معلومة.
                        4. اختم كل فقرة بمصدرها باستخدام الصيغة التالية:
                        📖 المصدر: [اسم القسم] - صفحة [رقم الصفحة]
                        5. إذا كانت المعلومة من عدة مصادر، اذكر كل المصادر.
                        6. احتفظ بنفس ترتيب النقاط كما وردت في النص الأصلي.
                        """
                    },
                    {"role": "user", "content": question}
                ]
            )
            
            response = make_response(jsonify({"answer": completion.choices[0].message.content}))
            response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            return response
            
        except Exception as openai_error:
            logger.error(f"OpenAI API error: {str(openai_error)}")
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
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
