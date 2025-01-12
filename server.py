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
import datetime

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
        self.section_hierarchy = defaultdict(list)  # New: Track section relationships

def preprocess_arabic_text(text):
    """Enhanced Arabic text preprocessing"""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize Arabic characters
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
        'ة': 'ه',
        'ى': 'ي',
        '‐': '-', '‑': '-', '–': '-', '—': '-'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def get_embedding(text, retries=3):
    """Get embedding from OpenAI API with improved error handling and retries"""
    text = preprocess_arabic_text(text)  # Preprocess text before embedding
    
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Final embedding attempt failed: {str(e)}")
                return None
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying...")
            continue

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    text1 = preprocess_arabic_text(text1)
    text2 = preprocess_arabic_text(text2)
    
    # Tokenize and create word sets
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def is_heading(paragraph):
    """Improved heading detection for Arabic text"""
    if not paragraph.text.strip():
        return False
    
    # Check style and formatting
    if paragraph.style and any(style in paragraph.style.name.lower() for style in 
        ['heading', 'title', 'header', 'العنوان', 'عنوان', 'رئيسي', 'فرعي']):
        return True
    
    # Check for bold formatting
    if paragraph.runs and any(run.bold for run in paragraph.runs):
        return True
    
    # Check for Arabic section markers and formatting
    text = paragraph.text.strip()
    return bool(text and (
        text.startswith('•') or 
        text.startswith('-') or 
        any(char in text for char in [':', '：', '：', '׃', '：']) or
        len(text.split()) <= 10  # Short phrases are likely headers
    ))

def process_text_chunk(text, max_length=300):  # Reduced chunk size
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
    """Load and process document with improved section handling"""
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Find document file
        docx_file = next((f for f in os.listdir(current_dir) 
                         if f.strip().endswith('arabic_file.docx')), None)
        
        if not docx_file:
            logger.error("Document not found")
            return None
            
        doc_path = os.path.join(current_dir, docx_file)
        logger.info(f"Loading document from: {doc_path}")
        
        doc = Document(doc_path)
        doc_content = DocumentContent()
        
        # Improved page detection
        page_marker_pattern = re.compile(r'(?:Page|صفحة|ص)\s*[:-]?\s*(\d+)')
        current_text = ""
        section_level = 0
        parent_sections = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            
            # Handle page numbers
            page_match = page_marker_pattern.search(text)
            if page_match:
                doc_content.current_page = int(page_match.group(1))
                continue
            
            # Handle sections and content
            if is_heading(paragraph):
                if current_text and doc_content.current_section:
                    # Process previous section's content
                    chunks = process_text_chunk(current_text)
                    for chunk in chunks:
                        doc_content.sections[doc_content.current_section].append({
                            'text': chunk,
                            'page': doc_content.current_page,
                            'parent_section': parent_sections[-1] if parent_sections else None
                        })
                    doc_content.section_text[doc_content.current_section] = current_text
                
                # Update section hierarchy
                if len(text.split()) <= 3:  # Likely a main section
                    section_level = 0
                    parent_sections = [text]
                else:
                    section_level += 1
                    if len(parent_sections) > section_level:
                        parent_sections = parent_sections[:section_level]
                    parent_sections.append(text)
                
                doc_content.current_section = text
                doc_content.section_hierarchy[text] = parent_sections[:-1]
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
                    'page': doc_content.current_page,
                    'parent_section': parent_sections[-1] if parent_sections else None
                })
            doc_content.section_text[doc_content.current_section] = current_text
        
        # Handle embeddings
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                doc_content.content = pickle.load(f)
            doc_content.index = faiss.read_index(INDEX_PATH)
            logger.info("Loaded existing embeddings and FAISS index")
        else:
            embeddings_list = []
            for section, chunks in doc_content.sections.items():
                for chunk in chunks:
                    doc_content.content.append({
                        'text': chunk['text'],
                        'section': section,
                        'page': chunk['page'],
                        'parent_section': chunk.get('parent_section')
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

def find_relevant_content(question, top_k=5):
    """Enhanced content search with semantic matching"""
    try:
        if not DOC_PROCESSOR:
            return []
        
        # Preprocess question
        processed_question = preprocess_arabic_text(question)
        question_embedding = get_embedding(processed_question)
        if not question_embedding:
            return []
        
        # Get initial candidates
        question_vector = np.array([question_embedding], dtype=np.float32)
        k_candidates = min(top_k * 6, len(DOC_PROCESSOR.content))
        distances, indices = DOC_PROCESSOR.index.search(question_vector, k_candidates)
        
        # Enhanced scoring
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            content = DOC_PROCESSOR.content[idx]
            
            # Calculate multiple relevance signals
            embedding_score = 1 - (dist / max(distances[0]))
            semantic_score = calculate_semantic_similarity(processed_question, content['text'])
            
            # Combined score with weights
            relevance_score = (
                embedding_score * 0.6 +
                semantic_score * 0.4
            )
            
            if relevance_score >= 0.2:  # Lower threshold for more candidates
                candidates.append({
                    'text': content['text'],
                    'section': content['section'],
                    'page': content['page'],
                    'relevance': relevance_score,
                    'parent_section': content.get('parent_section')
                })
        
        # Sort and select diverse results
        candidates.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Group by section and select best from each
        section_chunks = defaultdict(list)
        for chunk in candidates:
            section_chunks[chunk['section']].append(chunk)
        
        # Select diverse results
        final_chunks = []
        used_sections = set()
        
        # First, take highest scoring chunk from each section
        for section, chunks in section_chunks.items():
            if section not in used_sections and chunks:
                final_chunks.append(chunks[0])
                used_sections.add(section)
                
                # Also consider parent section
                if chunks[0].get('parent_section'):
                    used_sections.add(chunks[0]['parent_section'])
            
            if len(final_chunks) >= top_k:
                break
        
        # If we need more chunks, take second-best from sections
        if len(final_chunks) < top_k:
            for section, chunks in section_chunks.items():
                if len(chunks) > 1 and len(final_chunks) < top_k:
                    final_chunks.append(chunks[1])
        
        # Final sort by relevance
        final_chunks.sort(key=lambda x: x['relevance'], reverse=True)
        
        return final_chunks[:top_k]
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return []

# Initialize document processor
DOC_PROCESSOR = load_docx_content()

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
                "error": "عذراً، لم يتم تحميل الوثيقة بشكل صحيح."
            }), 500
        
        relevant_content = find_relevant_content(question, top_k=5)
        
        if not relevant_content:
            return jsonify({
                "answer": "عذرًا، لم أجد معلومات ذات صلة في التقرير."
            })

        # Build context with hierarchy information
        context_parts = []
        for i, item in enumerate(relevant_content, 1):
            context_parts.append(
                f"{i}. محتوى ذو صلة:\n"
                f"النص: {item['text']}\n"
                f"القسم: {item['section']}\n"
                f"القسم الرئيسي: {item.get('parent_section', 'لا يوجد')}\n"
                f"الصفحة: {item['page']}\n"
                f"درجة الصلة: {item['relevance']:.2f}\n"
                "---"
            )
        
        context = "\n".join(context_parts)

        # Enhanced system prompt
        system_prompt = f"""أنت مساعد متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية. استخدم المعلومات التالية للإجابة:

{context}

إرشادات التحليل والإجابة:
1. قم بتحليل جميع النصوص المقدمة بعناية
2. ابحث عن الروابط المنطقية بين المعلومات من مختلف الأقسام
3. رتب المعلومات حسب الأهمية والصلة بالسؤال
4. اذكر الأرقام والإحصائيات بدقة
5. اشر إلى مصدر كل معلومة (القسم والصفحة)

إذا كانت المعلومات غير كافية:
1. وضح المعلومات المتوفرة
2. اشرح أين قد تكون المعلومات الإضافية
3. اقترح إعادة صياغة السؤال إذا كان ذلك مفيداً"""

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            
            answer = completion.choices[0].message.content

            response_data = {
                "answer": answer,
                "debug_info": {
                    "chunks_used": len(relevant_content),
                    "relevance_scores": [
                        {
                            "score": f"{chunk['relevance']:.3f}",
                            "section": chunk['section'],
                            "page": chunk['page']
                        } 
                        for chunk in relevant_content
                    ],
                    "sections_used": list(set(chunk['section'] for chunk in relevant_content))
                }
            }
            
            response = make_response(jsonify(response_data))
            response.headers.add('Access-Control-Allow-Origin', 
                               'https://superlative-belekoy-1319b4.netlify.app')
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
    response.headers.add('Access-Control-Allow-Origin', 
                        'https://superlative-belekoy-1319b4.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with diagnostics"""
    try:
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
        
        return jsonify(health_info), 200
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Document path: {DOCUMENT_PATH}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )
