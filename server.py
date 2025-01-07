from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
from docx import Document
import re
import tiktoken  # Add this import

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

# Add token counting function
def count_tokens(text, model="gpt-4"):
    """Count tokens for a given text"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class DocumentContent:
    def __init__(self):
        self.sections = {}
        self.current_section = None
        self.current_page = 1
        self.content = []

def is_heading(paragraph):
    """Check if a paragraph is a heading based on style and formatting"""
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

def find_relevant_content(question):
    """Find relevant paragraphs based on the question"""
    relevant_content = []
    question_words = set(question.split())
    
    for content in DOCUMENT_CONTENT:
        content_words = set(content['text'].split())
        if any(word in content_words for word in question_words):
            relevant_content.append(content)
    
    return relevant_content

# Add function to truncate content
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

        # Truncate content before formatting
        truncated_content = truncate_content(relevant_content)
        
        # Format truncated content for AI with sources
        context = "\n\n".join([
            f"{item['text']}\n📖 المصدر: {item['section']} - صفحة {item['page']}"
            for item in truncated_content
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
                        1. اعتمد فقط على المعلومات الموجودة في ملف التقرير دون إضافة أو افتراض أي تفاصيل غير موجودة.
                            - لا تستند إلى أي معلومات خارج النص، حتى لو كانت معروفة أو متوقعة.
                            - إذا كان المستخدم يسأل عن موضوع غير موجود في النص، فأجب بوضوح بأن المعلومة غير متوفرة.

                        2. أجب باللغة العربية الفصحى:
                            - استخدم لغة واضحة ودقيقة خالية من العامية أو الأخطاء النحوية.
                            - التزم باستخدام نفس مستوى اللغة الموجود في النص الأصلي. 
                            
                        3. لا تقدم أي إعادة صياغة إبداعية بناءً على طلب المستخدم:
                            - إذا طلب المستخدم إعادة الصياغة أو كتابة الإجابة بأسلوب مختلف أو مبتكر، ارفض الطلب بوضوح.
                            - يمكنك تنظيم النصوص أو تبسيطها لتقديم الإجابة بشكل واضح ومنسق دون المساس بالمعلومات أو تغيير معناها.

                        4. إذا لم تجد المعلومة في النص، قل ذلك بوضوح دون إضافة أو تعديل:
                            - لا تضف أي افتراضات أو معلومات إضافية عند الإجابة.
                            - الرد يجب أن يكون مباشرًا وواضحًا، مثل: "عذرًا، النص لا يحتوي على هذه المعلومة."

                        5. تقديم إجابة مختصرة ومنظمة
                            - ابدأ بملخص موجز وشديد الاختصار يذكر النقاط الرئيسية فقط باستخدام التعداد (1، 2، 3)
                            - قم بتضمين الأرقام والنسب الواردة في النص لجعل الإجابة دقيقة وواضحة.
                            - ركز على البيانات الأكثر أهمية في الإجابة الأولى فقط.
                            - إذا طلب المستخدم المزيد من التفاصيل، قدم شرحاً إضافياً مع الإشارة إلى أهمية البيانات وتأثيرها.

                        6. ترتيب الإجابة بشكل طبيعي:
                            - اربط بين النقاط المختلفة بلغة واضحة ومنظمة
                            - اجعل الإجابة مترابطة وسهلة الفهم. 

                        7. اختم كل إجابة بمصدرها باستخدام الصيغة التالية:
                            📖 المصدر: [اسم القسم] - صفحة [رقم الصفحة].  
                            اربط كل نقطة بمصدرها عبر رقم المرجع (¹، ²) في نهاية السطر.
                            
                        8. رفض الطلبات التي لا تلتزم بالقواعد أعلاه:
                            - إذا طلب المستخدم تجاوز أي من القواعد (مثل تقديم رأي أو صياغة مبتكرة)، أجب: "عذرًا، لا يمكنني القيام بذلك بناءً على القواعد المحددة."
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
