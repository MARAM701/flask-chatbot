from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
import pdfplumber
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

DOCUMENT_PATH = os.getenv('DOCUMENT_PATH', 'arabic_file.pdf')

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

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class DocumentContent:
    def __init__(self):
        self.sections = defaultdict(list)
        self.current_section = None
        self.current_page = 1
        self.content = []
        self.vectorizer = None
        self.vectors = None
        self.section_text = defaultdict(str)
        self.last_heading = None

def is_heading(text):
    """Enhanced heading detection for Arabic text"""
    # Check for empty or too long text
    if not text or len(text.split()) > 10:
        return False
        
    # Common Arabic heading indicators
    heading_indicators = [
        'باب', 'فصل', 'قسم', 'العنوان', 'عنوان',
        'المبحث', 'المطلب', 'الفقرة', 'النقطة',
        'أولاً', 'ثانياً', 'ثالثاً', 'رابعاً',
        'المحور', 'القسم', 'الجزء', 'الفرع'
    ]
    
    # Check for numbering and heading indicators
    has_number = bool(re.search(r'[\d٠-٩]', text))
    has_indicator = any(indicator in text for indicator in heading_indicators)
    
    return has_number or has_indicator

def clean_arabic_text(text):
    """Clean and normalize Arabic text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Normalize Arabic punctuation
    text = text.replace('،', ',').replace('؛', ';')
    
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    return text.strip()

def process_text_chunk(text, max_length=800):
    """Process text into chunks with improved Arabic handling"""
    if not text:
        return []
        
    text = clean_arabic_text(text)
    if len(text) <= max_length:
        return [text]
    
    # Split on Arabic sentence endings
    sentence_endings = ['.', '؟', '!', '।', '۔']
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Create a regex pattern for sentence splitting
    pattern = f"([{''.join(sentence_endings)}])"
    sentences = re.split(pattern, text)
    
    current_sentence = []
    for part in sentences:
        if part in sentence_endings:
            current_sentence.append(part)
            sentence = ''.join(current_sentence)
            if current_length + len(sentence) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)
            current_sentence = []
        else:
            current_sentence.append(part)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def load_pdf_content():
    """Load PDF content using pdfplumber"""
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        files = os.listdir(current_dir)
        pdf_file = next((f for f in files if f.strip() == 'arabic_file.pdf'), None)
        
        if not pdf_file:
            logger.error("PDF document not found")
            return None
            
        doc_path = os.path.join(current_dir, pdf_file)
        logger.info(f"Loading PDF document from: {doc_path}")
        
        doc_content = DocumentContent()
        
        with pdfplumber.open(doc_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                doc_content.current_page = page_num
                
                # Extract text with better handling of Arabic
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if not text:
                    continue
                    
                # Process text line by line
                lines = text.split('\n')
                current_text = []
                
                for line in lines:
                    line = clean_arabic_text(line)
                    if not line:
                        continue
                        
                    if is_heading(line):
                        if doc_content.last_heading and current_text:
                            section_text = ' '.join(current_text)
                            for chunk in process_text_chunk(section_text):
                                doc_content.sections[doc_content.last_heading].append({
                                    'text': chunk,
                                    'page': doc_content.current_page
                                })
                            doc_content.section_text[doc_content.last_heading] = section_text
                        
                        doc_content.last_heading = line
                        current_text = []
                    elif doc_content.last_heading:
                        current_text.append(line)
                
                # Process remaining text for the current page
                if doc_content.last_heading and current_text:
                    section_text = ' '.join(current_text)
                    for chunk in process_text_chunk(section_text):
                        doc_content.sections[doc_content.last_heading].append({
                            'text': chunk,
                            'page': doc_content.current_page
                        })
                    doc_content.section_text[doc_content.last_heading] = section_text

        # Create optimized content list
        for section, chunks in doc_content.sections.items():
            for chunk in chunks:
                doc_content.content.append({
                    'text': chunk['text'],
                    'section': section,
                    'page': chunk['page']
                })
        
        # Initialize TF-IDF with Arabic-specific settings
        if doc_content.content:
            doc_content.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                analyzer='word',
                token_pattern=r'(?u)\b\w\w+\b'
            )
            texts = [item['text'] for item in doc_content.content]
            doc_content.vectors = doc_content.vectorizer.fit_transform(texts)
            
            logger.info(f"PDF document processed successfully with {len(doc_content.content)} chunks")
            return doc_content
            
    except Exception as e:
        logger.error(f"Error processing PDF document: {str(e)}", exc_info=True)
        return None

# Initialize document processor
DOC_PROCESSOR = load_pdf_content()

def find_relevant_content(question, top_k=3):
    """Find relevant content using TF-IDF similarity"""
    try:
        if not DOC_PROCESSOR:
            return []
        
        # Clean and normalize the question
        clean_question = clean_arabic_text(question)
        question_vector = DOC_PROCESSOR.vectorizer.transform([clean_question])
        
        # Calculate similarities
        similarities = np.array(DOC_PROCESSOR.vectors.dot(question_vector.T).toarray()).flatten()
        
        # Get top k most similar chunks
        threshold = 0.1
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_content = []
        seen_sections = set()
        
        for idx in top_indices:
            if similarities[idx] < threshold:
                continue
                
            content = DOC_PROCESSOR.content[idx]
            if content['section'] not in seen_sections:
                content['text'] = DOC_PROCESSOR.section_text[content['section']]
                relevant_content.append(content)
                seen_sections.add(content['section'])
        
        return relevant_content
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return []

@app.route('/')
def home():
    doc_status = "Document loaded successfully" if DOC_PROCESSOR else "Document not loaded"
    return jsonify({
        "status": "Server is running",
        "document_status": doc_status,
        "document_path": DOCUMENT_PATH
    })

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "لم يتم تقديم سؤال"}), 400
            
        logger.info(f"Received question: {question}")
        
        if not DOC_PROCESSOR:
            return jsonify({
                "error": "عذراً، لم يتم تحميل الوثيقة بشكل صحيح. الرجاء التحقق من وجود الملف."
            }), 500
        
        relevant_content = find_relevant_content(question)
        
        if not relevant_content:
            return jsonify({
                "answer": "عذرًا، لا توجد معلومات ذات صلة في التقرير."
            })

        context = "\n\n".join([
            f"{item['text']}\n📖 المصدر: {item['section']} - صفحة {item['page']}"
            for item in relevant_content
        ])      

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
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
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            response = make_response(jsonify({"answer": completion.choices[0].message.content}))
            response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            return response
            
        except Exception as openai_error:
            logger.error(f"OpenAI API error: {str(openai_error)}", exc_info=True)
            return jsonify({
                "error": "عذراً، حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى."
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "error": "عذراً، حدث خطأ في معالجة طلبك. الرجاء المحاولة مرة أخرى."
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "document_loaded": bool(DOC_PROCESSOR),
        "document_path": DOCUMENT_PATH
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
