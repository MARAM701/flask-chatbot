from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
from docx import Document
import re
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import gc

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

# Initialize the model with a smaller multilingual model that handles Arabic well
try:
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    model.to('cpu')  # Ensure model is on CPU
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    # Fallback to an even smaller model if the first one fails
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model.to('cpu')

class DocumentContent:
    def __init__(self):
        self.sections = {}
        self.current_section = None
        self.current_page = 1
        self.content = []
        self.embeddings = None

    def compute_embeddings(self, batch_size=4):
        """Compute embeddings in batches to save memory"""
        texts = [item['text'] for item in self.content]
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Compute embeddings for batch
            with torch.no_grad():
                embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                all_embeddings.append(embeddings.cpu())

            # Clear memory
            gc.collect()

        # Concatenate all batches
        self.embeddings = torch.cat(all_embeddings, dim=0)
        
        # Final memory cleanup
        gc.collect()
        return self.embeddings

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
        
        # Compute embeddings in batches
        doc_content.compute_embeddings()
        logger.info("Successfully loaded document content and computed embeddings")
        return doc_content
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        return None

# Load report content when server starts
DOCUMENT_CONTENT = load_docx_content()

def find_relevant_content(question, top_k=3):
    """Find relevant paragraphs using semantic search"""
    try:
        # Encode the question
        with torch.no_grad():
            question_embedding = model.encode(question, convert_to_tensor=True, show_progress_bar=False)

        # Calculate similarities using pytorch utility
        cos_scores = util.pytorch_cos_sim(question_embedding, DOCUMENT_CONTENT.embeddings)[0]
        
        # Get top_k most similar indices
        top_results = torch.topk(cos_scores, k=min(top_k, len(DOCUMENT_CONTENT.content)))
        
        # Get the relevant content
        relevant_content = [DOCUMENT_CONTENT.content[idx] for idx in top_results.indices]
        
        # Memory cleanup
        gc.collect()
        
        return relevant_content
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []

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
        
        # Find relevant content using semantic search
        relevant_content = find_relevant_content(question)
        
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
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية لعام 2023. استخدم المعلومات التالية للإجابة على الأسئلة:

                        {context}

                        قواعد مهمة:
                        1. اعتمد فقط على المعلومات الموجودة في ملف التقرير دون إضافة أو افتراض أي تفاصيل غير موجودة.
                        2. أجب باللغة العربية الفصحى.
                        3. لا تقدم أي إعادة صياغة إبداعية.
                        4. قدم إجابة مختصرة ومنظمة.
                        5. اختم كل إجابة بمصدرها."""
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
