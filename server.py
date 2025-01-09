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

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
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
    """Get embedding from OpenAI API with improved error handling"""
    try:
        # Add retry logic for reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Embedding attempt {attempt + 1} failed, retrying...")
                continue
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def is_heading(paragraph):
    """Improved heading detection for Arabic and English text"""
    # Check for explicit style names
    if paragraph.style and any(style in paragraph.style.name.lower() for style in 
        ['heading', 'title', 'header', 'العنوان', 'عنوان', 'رئيسي', 'فرعي']):
        return True
    
    # Check for bold formatting
    if paragraph.runs and any(run.bold for run in paragraph.runs):
        return True
    
    # Additional checks for Arabic headings
    text = paragraph.text.strip()
    if text and (text.startswith('•') or text.startswith('-') or 
                any(char in text for char in [':', '：', '：', '׃', '：'])):
        return True
        
    return False

def process_text_chunk(text, max_length=400):  # Reduced chunk size for better context
    """Split text into smaller chunks with improved Arabic text handling"""
    if len(text) <= max_length:
        return [text]
    
    # Enhanced sentence splitting for Arabic
    sentence_endings = '[.!?।؟۔؛،]+'
    sentences = re.split(f'({sentence_endings})', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        
        if not sentence:
            i += 2
            continue
        
        # Preserve sentence endings
        if i + 1 < len(sentences) and re.match(sentence_endings, sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        # Create new chunk if current one is too large
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
    """Load and process document with improved error handling and logging"""
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
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
        
        # Improved page marker detection
        page_marker_pattern = re.compile(r'(?:Page|صفحة|ص)\s*[:-]?\s*(\d+)')
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
        
        # Handle embeddings
        embeddings_list = []
        
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
            
            if embeddings_list:
                dimension = len(embeddings_list[0])
                doc_content.index = faiss.IndexFlatL2(dimension)
                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                doc_content.index.add(embeddings_array)
                
                with open(EMBEDDINGS_PATH, 'wb') as f:
                    pickle.dump(doc_content.content, f)
                faiss.write_index(doc_content.index, INDEX_PATH)
                logger.info("Created and saved new embeddings and FAISS index")
            else:
                logger.error("No embeddings generated")
                return None
        
        logger.info(f"Document processed successfully with {len(doc_content.content)} chunks")
        return doc_content
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return None

# Initialize document processor
DOC_PROCESSOR = load_docx_content()

def find_relevant_content(question, top_k=5):  # Increased from 3 to 5
    """Find relevant content using improved search and context handling"""
    try:
        if not DOC_PROCESSOR:
            return []
        
        question_embedding = get_embedding(question)
        if not question_embedding:
            return []
        
        # Get more candidates for better coverage
        question_vector = np.array([question_embedding], dtype=np.float32)
        k_candidates = min(top_k * 4, len(DOC_PROCESSOR.content))
        distances, indices = DOC_PROCESSOR.index.search(question_vector, k_candidates)
        
        # Log search results
        logger.debug(f"\nSearch results for question: {question}")
        logger.debug("Raw candidates (idx | distance | text snippet):")
        for idx, dist in zip(indices[0], distances[0]):
            text_snippet = DOC_PROCESSOR.content[idx]['text'][:100]
            logger.debug(f"idx={idx}, dist={dist:.4f}, text={text_snippet}...")
        
        # Improved scoring
        max_dist = np.max(distances[0]) if len(distances[0]) > 0 else 1
        min_dist = np.min(distances[0]) if len(distances[0]) > 0 else 0
        dist_range = max_dist - min_dist
        
        if dist_range > 0:
            relevance_scores = 1 - ((distances[0] - min_dist) / dist_range)
        else:
            relevance_scores = np.ones_like(distances[0])
        
        relevant_chunks = []
        section_chunks = defaultdict(list)
        
        # Group chunks by section
        for idx, score in zip(indices[0], relevance_scores):
            if score < 0.3:  # Lowered threshold from 0.5
                continue
                
            content = DOC_PROCESSOR.content[idx]
            section = content['section']
            
            section_chunks[section].append({
                'text': content['text'],  # Use chunk text instead of section text
                'page': content['page'],
                'relevance': score,
                'section': section
            })
        
        # Select best chunks from each section
        for section, chunks in section_chunks.items():
            chunks.sort(key=lambda x: x['relevance'], reverse=True)
            relevant_chunks.extend(chunks[:2])  # Take up to 2 best chunks per section
            
            if len(relevant_chunks) >= top_k * 2:  # Get more candidates
                break
        
        # Final sort and limit
        relevant_chunks.sort(key=lambda x: x['relevance'], reverse=True)
        relevant_chunks = relevant_chunks[:top_k]
        
        logger.debug("\nSelected chunks:")
        for chunk in relevant_chunks:
            logger.debug(f"Section: {chunk['section']}, Page: {chunk['page']}, Score: {chunk['relevance']:.4f}")
            logger.debug(f"Text: {chunk['text'][:100]}...")
        
        return relevant_chunks
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return []

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
        
        relevant_content = find_relevant_content(question, top_k=5)
        
        if not relevant_content:
            return jsonify({
                "answer": "عذرًا، لا توجد معلومات ذات صلة في التقرير."
            })

        # Enhanced context building
        context_parts = []
        for item in relevant_content:
            context_parts.append(
                f"النص: {item['text']}\n"
                f"القسم: {item['section']}\n"
                f"الصفحة: {item['page']}\n"
                f"درجة الصلة: {item['relevance']:.2f}\n"
                "---"
            )
        
        context = "\n".join(context_parts)

        # Enhanced system prompt
        system_prompt = f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية. يرجى استخدام المعلومات التالية للإجابة:

{context}

إرشادات مهمة:
1. ابحث عن المعلومات في كل النصوص المقدمة، حتى لو كانت مصاغة بشكل مختلف.
2. إذا وجدت المعلومة في أكثر من مكان، قم بدمجها بشكل منطقي.
3. إذا كانت المعلومة غير واضحة أو غير مباشرة، اشرح ما وجدته وأين يمكن أن تكون المعلومة غير مكتملة.
4. اذكر دائماً مصدر المعلومة (القسم والصفحة) في نهاية إجابتك.
5. إذا كنت غير متأكد، قل ذلك بوضوح واذكر السبب.

عند الإجابة:
- اجمع المعلومات من جميع النصوص ذات الصلة
- ابدأ بالنقاط الرئيسية ثم التفاصيل
- اذكر الأرقام والإحصائيات بدقة 
- اربط المعلومات بشكل منطقي
- اختم بمصادر المعلومات المستخدمة"""

        try:
            # Use GPT-3.5-turbo-16k for larger context
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                temperature=0.2,  # Lower temperature for more consistent answers
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            
            answer = completion.choices[0].message.content

            # Add debug information to response
            response_data = {
                "answer": answer,
                "debug_info": {
                    "chunks_used": len(relevant_content),
                    "relevance_scores": [f"{chunk['relevance']:.3f}" for chunk in relevant_content],
                    "sections_used": list(set(chunk['section'] for chunk in relevant_content))
                }
            }
            
            response = make_response(jsonify(response_data))
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
    """Build CORS preflight response"""
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with detailed diagnostics"""
    try:
        # Basic health information
        health_info = {
            "status": "healthy",
            "document_loaded": bool(DOC_PROCESSOR),
            "document_path": DOCUMENT_PATH,
            "chunks_count": len(DOC_PROCESSOR.content) if DOC_PROCESSOR else 0,
            "sections_count": len(DOC_PROCESSOR.sections) if DOC_PROCESSOR else 0,
            "embeddings_file_exists": os.path.exists(EMBEDDINGS_PATH),
            "index_file_exists": os.path.exists(INDEX_PATH),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add memory usage info if psutil is available
        try:
            import psutil
            process = psutil.Process()
            health_info["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
            health_info["cpu_percent"] = process.cpu_percent()
        except ImportError:
            pass

        return jsonify(health_info), 200
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Add necessary imports
import datetime

if __name__ == '__main__':
    # Configure server
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Log startup information
    logger.info(f"Starting server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Document path: {DOCUMENT_PATH}")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True  # Enable threading for better concurrent performance
    )
