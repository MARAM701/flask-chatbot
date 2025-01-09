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
import torch
from transformers import convert_graph_to_onnx, AutoTokenizer
import onnxruntime as ort
import datetime

logging.basicConfig(
   level=logging.DEBUG,
   format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger('server')

DOCUMENT_PATH = os.getenv('DOCUMENT_PATH', 'arabic_file.docx')
EMBEDDINGS_PATH = 'embeddings.pkl'
INDEX_PATH = 'faiss_index.bin'

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'qa_model.onnx'

if not MODEL_PATH.exists():
   MODEL_DIR.mkdir(parents=True, exist_ok=True)
   from transformers import AutoModelForQuestionAnswering
   model = AutoModelForQuestionAnswering.from_pretrained("ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA")
   convert_graph_to_onnx.convert(
       framework="pt",
       model=model,
       output=MODEL_PATH,
       pipeline_name="question-answering",
       opset=12,
       clean_dir=True
   )

tokenizer = AutoTokenizer.from_pretrained("ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA")
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session = ort.InferenceSession(str(MODEL_PATH), session_options)

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

def get_extractive_answer(question, context):
   try:
       inputs = tokenizer(
           question, 
           context, 
           return_tensors="pt", 
           max_length=384, 
           truncation=True
       )
       outputs = session.run(None, {
           "input_ids": inputs["input_ids"].numpy(),
           "attention_mask": inputs["attention_mask"].numpy()
       })
       answer_start = outputs[0].argmax()
       answer_end = outputs[1].argmax()
       return tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
   except Exception as e:
       logger.error(f"ONNX inference error: {e}")
       return "تعذر استخراج الإجابة"

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
           return jsonify({"error": "عذراً، لم يتم تحميل الوثيقة بشكل صحيح."}), 500
       
       relevant_content = find_relevant_content(question, top_k=3)
       if not relevant_content:
           return jsonify({"answer": "عذرًا، لا توجد معلومات ذات صلة في التقرير."})

       combined_context = "\n\n".join([f"{chunk['text']}" for chunk in relevant_content])
       extractive_answer = get_extractive_answer(question, combined_context)

       system_prompt = f"""
       أنت مساعد ذكي متخصص في الإجابة عن الأسئلة المتعلقة بالتقرير السنوي.
       استخدم النص المستخرج التالي كإجابة أولية وقدم شرحًا مفصلًا بلغة عربية فصحى:
       
       الإجابة المستخرجة: "{extractive_answer}"
       """
       
       completion = client.chat.completions.create(
           model="gpt-4",
           messages=[
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": question}
           ]
       )

       detailed_answer = completion.choices[0].message.content

       response_data = {
           "extractive_answer": extractive_answer,
           "detailed_answer": detailed_answer,
           "debug_info": {
               "context_used": combined_context[:300],
               "relevant_chunks_count": len(relevant_content),
               "chunks": [chunk["text"][:100] for chunk in relevant_content]
           }
       }
       
       response = make_response(jsonify(response_data))
       response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
       response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
       response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
       return response
           
   except Exception as e:
       logger.error(f"Error processing request: {str(e)}", exc_info=True)
       return jsonify({"error": "عذراً، حدث خطأ في معالجة طلبك. الرجاء المحاولة مرة أخرى."}), 500

def _build_cors_preflight_response():
   response = make_response()
   response.headers.add('Access-Control-Allow-Origin', 'https://superlative-belekoy-1319b4.netlify.app')
   response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
   response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
   return response

@app.route('/health', methods=['GET'])
def health_check():
   return jsonify({"status": "healthy", "document_loaded": bool(DOC_PROCESSOR)}), 200

if __name__ == '__main__':
   port = int(os.getenv('PORT', 5000))
   debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
   logger.info(f"Starting server on port {port}")
   app.run(host='0.0.0.0', port=port, debug=debug_mode)
