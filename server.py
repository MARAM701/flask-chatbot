from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
from docx import Document
import re
from pathlib import Path
import numpy as np
from collections import defaultdict
import faiss
import pickle

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

DOCUMENT_PATH = os.getenv('DOCUMENT_PATH', 'arabic_file.docx')
EMBEDDINGS_PATH = 'embeddings.pkl'
INDEX_PATH = 'faiss_index.bin'

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
        self.embeddings = None
        self.index = None
        self.section_text = defaultdict(str)

def get_embedding(text):
    """Get embedding from OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def is_heading(paragraph):
    if paragraph.style and any(style in paragraph.style.name.lower() for style in ['heading', 'title', 'header', 'العنوان', 'عنوان']):
        return True
    
    if paragraph.runs and paragraph.runs[0].bold:
        return True
        
    return False

def process_text_chunk(text, max_length=1000):
    """Split text into chunks if too long"""
    if len(text) <= max_length:
        return [text]
    
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        if current_length + len(sentence) > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def load_docx_content():
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Find the docx file ignoring leading/trailing whitespace
        files = os.listdir(current_dir)
        docx_file = next((f for f in files if f.strip() == 'arabic_file.docx'), None)
        if not docx_file:
            docx_file = next((f for f in files if f.strip().endswith('arabic_file.docx')), None)
        
        if not docx_file:
            logger.error("Document not found")
            return None
            
        doc_path = os.path.join(current_dir, docx_file)
        logger.info(f"Loading document from: {doc_path}")
        
        doc = Document(doc_path)
        doc_content = DocumentContent()
        
        page_marker_pattern = re.compile(r'Page\s+(\d+)')
        current_text = ""
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            page_match = page_marker_pattern.search(text)
            if page_match:
                doc_content.current_page = int(page_match.group(1))
                continue
            
            if is_heading(paragraph):
                # Process previous section
                if current_text and doc_content.current_section:
                    chunks = process_text_chunk(current_text)
                    for chunk in chunks:
                        doc_content.sections[doc_content.current_section].append({
                            'text': chunk,
                            'page': doc_content.current_page
                        })
                    doc_content.section_text[doc_content.current_section] = current_text
                
                doc_content.current_section = text
                current_text = ""
                continue
            
            if doc_content.current_section:
                current_text += " " + text
        
        # Process final section
        if current_text and doc_content.current_section:
            chunks = process_text_chunk(current_text)
            for chunk in chunks:
                doc_content.sections[doc_content.current_section].append({
                    'text': chunk,
                    'page': doc_content.current_page
                })
            doc_content.section_text[doc_content.current_section] = current_text
        
        # Create flat content list and generate embeddings
        embeddings_list = []
        
        # Try to load existing embeddings
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                doc_content.content = pickle.load(f)
            doc_content.index = faiss.read_index(INDEX_PATH)
            logger.info("Loaded existing embeddings and FAISS index")
        else:
            for section, chunks in doc_content.sections.items():
                for chunk in chunks:
                    doc_content.content.append({
                        'text': chunk['text'],
                        'section': section,
                        'page': chunk['page']
                    })
                    embedding = get_embedding(chunk['text'])
                    if embedding:
                        embeddings_list.append(embedding)
            
            # Create FAISS index
            dimension = len(embeddings_list[0])
            doc_content.index = faiss.IndexFlatL2(dimension)
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            doc_content.index.add(embeddings_array)
            
            # Save embeddings and index
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(doc_content.content, f)
            faiss.write_index(doc_content.index, INDEX_PATH)
            logger.info("Created and saved new embeddings and FAISS index")
        
        logger.info(f"Document processed successfully with {len(doc_content.content)} chunks")
        return doc_content
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return None

# Initialize document processor
DOC_PROCESSOR = load_docx_content()

def find_relevant_content(question, top_k=3):
    """Find relevant content using FAISS similarity search"""
    try:
        if not DOC_PROCESSOR:
            return []
        
        # Get question embedding
        question_embedding = get_embedding(question)
        if not question_embedding:
            return []
        
        # Search similar content
        question_vector = np.array([question_embedding], dtype=np.float32)
        distances, indices = DOC_PROCESSOR.index.search(question_vector, top_k)
        
        relevant_content = []
        seen_sections = set()
        
        for idx in indices[0]:
            content = DOC_PROCESSOR.content[idx]
            if content['section'] not in seen_sections:
                # Get full section text
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
                        1.  **اعتمد على النص فقط:** أجب فقط من النص دون افتراضات. إذا غابت المعلومة، قل: "عذرًا، النص لا يحتوي على هذه المعلومة."

                        2. **استخدم العربية الفصحى:** استخدم لغة فصحى واضحة خالية من الأخطاء والعامية.
                            
                        3. **تجنب الصياغة الإبداعية:** لا تعيد كتابة النص بأسلوب مختلف، بل قم بتنسيق الإجابة فقط عند الحاجة.
                            
                        4. **الإيجاز والتنظيم:** ابدأ بملخص يتضمن الأرقام والنسب المهمة. أضف التفاصيل عند الطلب.

                        5. **الانسجام والترابط:** اجعل الأفكار مترابطة وسهلة الفهم.

                        6. **التصريح عند غياب المعلومة:** إذا لم توجد المعلومة، قلها بوضوح.

                        7. اختم كل إجابة بمصدرها باستخدام الصيغة التالية:
                            📖 المصدر: [اسم القسم] - صفحة [رقم الصفحة].  
                            اربط كل نقطة بمصدرها عبر رقم المرجع (¹، ²) في نهاية السطر.
                            
                        8. **الرفض عند مخالفة القواعد:** إذا طلب تجاوز القواعد، قل: "عذرًا، لا يمكنني القيام بذلك بناءً على القواعد المحددة."
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
            logger.error(f"OpenAI API error: {str(openai_error)}", exc_info=True)
            return jsonify({
                "error": "عذراً، حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى."
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
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
        "document_loaded": bool(DOC_PROCESSOR),
        "document_path": DOCUMENT_PATH
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
