from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
from docx import Document
import re
import tiktoken

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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('server')

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class DocumentContent:
    def __init__(self):
        self.sections = {}
        self.current_section = None
        self.current_page = 1
        self.content = []

def is_heading(paragraph):
    if paragraph.style and any(style in paragraph.style.name for style in ['Heading', 'Title', 'Header', 'العنوان', 'عنوان']):
        return True
    if paragraph.runs and paragraph.runs[0].bold:
        return True
    return False

def load_docx_content():
    try:
        doc = Document('arabic_file.docx')
        doc_content = DocumentContent()
        page_marker_pattern = re.compile(r'Page\s+(\d+)')
        logger.info("Starting document processing")
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            logger.info(f"Processing: {text[:50]}... | Style: {paragraph.style.name if paragraph.style else 'No style'}")
            
            page_match = page_marker_pattern.search(text)
            if page_match:
                doc_content.current_page = int(page_match.group(1))
                continue
            
            if is_heading(paragraph):
                doc_content.current_section = text
                logger.info(f"Found header: {text}")
                continue
            
            if doc_content.current_section:
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

DOCUMENT_CONTENT = load_docx_content()

def calculate_relevance_score(question, content_text):
    """Calculate relevance score between question and content"""
    question_words = set(question.split())
    content_words = set(content_text.split())
    matches = sum(1 for word in question_words if word in content_words)
    return matches / len(question_words) if question_words else 0

def find_relevant_content(question):
    """Find and score relevant paragraphs based on the question"""
    scored_content = []
    
    for content in DOCUMENT_CONTENT:
        score = calculate_relevance_score(question, content['text'])
        if score > 0:
            scored_content.append({
                **content,
                'relevance_score': score
            })
    
    # Sort by relevance score
    scored_content.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_content

def truncate_content(relevant_content, max_tokens=6000):
    """Truncate content to stay within token limits"""
    truncated_content = []
    total_tokens = 0
    
    for item in relevant_content:
        content_text = f"{item['text']}\n📖 المصدر: {item['section']} - صفحة {item['page']}"
        tokens = count_tokens(content_text)
        
        if total_tokens + tokens <= max_tokens:
            truncated_content.append(item)
            total_tokens += tokens
        else:
            break
    
    return truncated_content

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
        
        relevant_content = find_relevant_content(question)
        
        if not relevant_content:
            return jsonify({
                "answer": "عذرًا، لا توجد معلومات ذات صلة في التقرير."
            })

        truncated_content = truncate_content(relevant_content)
        
        # Two-step process: First summarize, then format
        context = "\n\n".join([
            f"{item['text']}\n📖 المصدر: {item['section']} - صفحة {item['page']}"
            for item in truncated_content
        ])

        try:
            # First step: Generate initial summary
            summary_completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """أنت محلل متخصص في تلخيص المعلومات. قم بتحليل المحتوى وتقديم:
                        - ملخص مركز للنقاط الأساسية
                        - ترتيب المعلومات حسب الأهمية
                        - ربط كل معلومة بمصدرها"""
                    },
                    {
                        "role": "user",
                        "content": f"السؤال: {question}\n\nالمحتوى للتحليل:\n{context}"
                    }
                ]
            )

            # Second step: Format the final response
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية لعام 2023. قم بتنسيق وتنظيم هذا الملخص في إجابة نهائية:

                        {summary_completion.choices[0].message.content}

                        قواعد المهمة:
                        1. قدم إجابة مختصرة ومركزة.
                        2. رتب المعلومات حسب الأهمية.
                        3. اذكر المصادر بالصيغة المطلوبة.
                        4. لا تضف أي معلومات غير موجودة في النص."""
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
