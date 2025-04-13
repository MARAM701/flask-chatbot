from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
import re
import json
import numpy as np
from openai import OpenAI
from numpy.linalg import norm

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

# Update to handle multiple JSON files dynamically
JSON_FILE_PATHS = os.getenv('JSON_FILE_PATHS', 'report_2016.json,report_2017.json,report_2018.json,report_2019.json,report_2020.json,report_2021.json,report_2022.json,report_2023.json')
# Convert the comma-separated string into a list
JSON_FILES = [path.strip() for path in JSON_FILE_PATHS.split(',')]
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

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (norm(a) * norm(b))

def create_embedding(text):
    """Create embedding for a given text using OpenAI's API"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        return None

def load_json_data():
    """Load and preprocess all JSON files at server startup"""
    global REPORT_DATA
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        REPORT_DATA = []  # Reset the global data
        
        # Process each JSON file in the list
        for json_file in JSON_FILES:
            json_path = os.path.join(current_dir, json_file)
            logger.info(f"Loading JSON file from: {json_path}")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    report_data = json.load(file)
                
                logger.info(f"Successfully loaded {len(report_data)} sections from {json_file}")
                REPORT_DATA.extend(report_data)  # Add to global data
                
            except FileNotFoundError:
                logger.warning(f"Report file {json_file} not found. Skipping.")
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON in {json_file}. Skipping this file.")
        
        logger.info(f"Combined {len(REPORT_DATA)} total sections from all reports")
        
        # Add embeddings to each section if not already present
        for section in REPORT_DATA:
            if "embedding" not in section:
                logger.info(f"Creating embedding for section: {section.get('section_header', 'Unknown')}")
                section_content = section.get('content', '')
                section["embedding"] = create_embedding(section_content)
        
        if not REPORT_DATA:
            logger.error("No data was loaded from any report files.")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error loading JSON files: {str(e)}", exc_info=True)
        return False

def ask_gpt4(question, context):
    """Send the document and question to OpenAI GPT-4 API."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = """أنت مساعد متخصص في تحليل النصوص العربية والإجابة على الأسئلة بدقة عالية.

إذا كان المستخدم يرسل فقط تحية أو سلام (مثل "السلام عليكم" أو "مرحبا" أو "صباح الخير")، رد بتحية مناسبة ومحترمة متبوعة بعبارة مثل:
"كيف يمكنني مساعدتك اليوم؟ أنا هنا للإجابة على أسئلتك حول تقارير كاكست السنويه."

ولكن، إذا كان المستخدم يطرح سؤالاً محدداً، عليك أن تبحث في جميع الأقسام المتاحة وتجيب وفقاً للقواعد التالية:

1. إذا كانت المعلومات مأخوذة من قسم واحد فقط:
**الإجابة:** 
[إجابتك المبنية على النص] [1]

**النص الأصلي:**
"[أول 30 حرف من النص المقتبس]..."
    
📖 المصدر:
[اسم الملف] - [اسم القسم]

2. إذا كانت المعلومات مأخوذة من عدة أقسام:
**الإجابة:**
[إجابتك المبنية على النص مع رقم المرجع بعد كل معلومة]

**النص الأصلي:**
[1]: "[أول 30 حرف من النص المقتبس]..."
[2]: "[أول 30 حرف من النص المقتبس]..."
    
📖 المصادر:
[1]: [اسم الملف للقسم الأول] - [اسم القسم الأول]
[2]: [اسم الملف للقسم الثاني] - [اسم القسم الثاني]

3. إذا لم تجد المعلومة في النص، اكتب:
**الإجابة:** عذراً، لم أجد معلومات في النص تجيب على هذا السؤال.

4. التزم بالقواعد التالية:
- اعتمد فقط على المعلومات الموجودة في النص
- ابحث في جميع الأقسام قبل تقديم الإجابة
- أضف رقم المرجع [N] بعد كل معلومة مقتبسة
- اقتبس فقط أول 30 حرف من النص الأصلي متبوعة بثلاث نقاط (...)
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
    """Format GPT response using the exact file_name and section_header from JSON data"""
    # Check if it's a "no information found" response
    if "عذراً، لم أجد معلومات" in gpt_response:
        return gpt_response
    
    # Check if it's a greeting response (containing common greeting responses or mentions of KACST)
    greeting_indicators = [
        "وعليكم السلام", "أهلاً", "مرحباً", "صباح النور", 
        "تقارير كاكست", "كيف يمكنني مساعدتك", "كاكست"
    ]
    
    if any(indicator in gpt_response for indicator in greeting_indicators):
        return gpt_response
    
    # Handle both single and multiple sources
    if "📖 المصدر:" in gpt_response:
        # Convert single source format to multiple source format
        gpt_response = gpt_response.replace("📖 المصدر:", "📖 المصادر:\n[1]:")
    
    # Find all section references in the format after "📖 المصادر:"
    if "📖 المصادر:" in gpt_response:
        parts = gpt_response.split("📖 المصادر:")
        pre_sources = parts[0]
        sources_section = parts[1]
        
        # Process each line in the sources section
        processed_lines = []
        for line in sources_section.strip().split('\n'):
            # Match reference pattern [N]: any text
            match = re.match(r'\[(\d+)\]:\s*(.*?)$', line.strip())
            if match:
                ref_num = match.group(1)
                section_text = match.group(2).strip()
                
                # Look for exact match in REPORT_DATA
                found = False
                for section in REPORT_DATA:
                    exact_match = f"{section.get('file_name')} - {section.get('section_header')}"
                    if exact_match == section_text:
                        processed_lines.append(f'[{ref_num}]: {section_text}')
                        found = True
                        break
                
                # Skip if no exact match found (no fallback logic)
                # We don't add anything to processed_lines if not found
            else:
                # Keep non-reference lines but only if they're not empty
                if line.strip():
                    processed_lines.append(line)
        
        # Reconstruct the response with updated sources
        gpt_response = pre_sources + "📖 المصادر:\n" + "\n".join(processed_lines)
    
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

    # Create embedding for the question
    question_embedding = create_embedding(question)
    
    if question_embedding:
        # Calculate similarity with all sections and get top 5
        similarities = []
        for i, section in enumerate(REPORT_DATA):
            section_embedding = section.get('embedding')
            if section_embedding:
                similarity = cosine_similarity(question_embedding, section_embedding)
                similarities.append((i, similarity))
        
        # Sort by similarity (descending) and take top 5
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_sections = similarities[:5]
        
        # Format context with only the top 5 most relevant sections
        context_parts = []
        for idx, _ in top_sections:
            section = REPORT_DATA[idx]
            file_name = section.get('file_name', '')
            section_header = section.get('section_header', '')
            content = section.get('content', '')
            
            context_parts.append(f"""
=== المصدر: {file_name} - القسم: {section_header} ===
{content}
=== نهاية القسم: {section_header} من المصدر: {file_name} ===
""")
        
        context = "\n\n".join(context_parts)
    else:
        # Fallback to using all sections if embedding fails
        logger.warning("Question embedding failed, using all sections as context")
        context_parts = []
        for section in REPORT_DATA:
            file_name = section.get('file_name', '')
            section_header = section.get('section_header', '')
            content = section.get('content', '')
            
            context_parts.append(f"""
=== المصدر: {file_name} - القسم: {section_header} ===
{content}
=== نهاية القسم: {section_header} من المصدر: {file_name} ===
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
        "json_files": JSON_FILES,
        "sections_count": len(REPORT_DATA),
        "sections_with_embeddings": sum(1 for section in REPORT_DATA if "embedding" in section)
    }), 200

# Load the JSON data when the server starts
load_json_data()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
