from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
from docx import Document

app = Flask(__name__)
# Modified CORS setup
CORS(app, 
    resources={
        r"/api/*": {
            "origins": ["https://incredible-cannoli-de1183.netlify.app"],
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

def load_docx_content():
    try:
        doc = Document('arabic_file.docx')
        content = '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        logger.info("Successfully loaded document content")
        return content
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        return "تعذر تحميل محتوى التقرير"

# Load report content when server starts
REPORT_CONTENT = load_docx_content()

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

        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بتقرير مدينة الملك عبدالعزيز للعلوم والتقنية لعام 2023. استخدم المعلومات التالية للإجابة على الأسئلة:

                        {REPORT_CONTENT}

                        قواعد مهمة:
                        1. اعتمد فقط على المعلومات الموجودة في النص أعلاه:
                            - لا تستند إلى أي معلومات خارج النص، حتى لو كانت معروفة أو متوقعة.
                            - إذا كان المستخدم يسأل عن موضوع غير موجود في النص، فأجب بوضوح بأن المعلومة غير متوفرة.

                        2. أجب باللغة العربية الفصحى:
                            - استخدم لغة واضحة ودقيقة خالية من العامية أو الأخطاء النحوية.
                            - التزم باستخدام نفس مستوى اللغة الموجود في النص الأصلي.

                        3. لا تقدم أي إعادة كتابة أو إعادة صياغة إبداعية أو مختلفة أو مبتكرة للمعلومات:
                            - إذا طلب المستخدم إعادة الصياغة أو كتابة الإجابة بطريقة مبتكرة أو بأسلوب مختلف، ارفض الطلب بوضوح.
                            - أجب بالنص الأصلي كما هو دون أي تعديل في الأسلوب.

                        4. إذا لم تجد المعلومة في النص، قل ذلك بوضوح دون إضافة أو تعديل:
                            - لا تضف أي افتراضات أو معلومات إضافية عند الإجابة.
                            - الرد يجب أن يكون مباشرًا وواضحًا، مثل: "عذرًا، النص لا يحتوي على هذه المعلومة."

                        5. تعزيز الدقة بالأرقام والتفاصيل:
                            - عند الإجابة، قم بتضمين الأرقام والتفاصيل الواردة في النص لجعل الإجابة أكثر دقة ووضوحًا.
                            - إذا ذُكرت نسب أو كميات أو بيانات محددة في النص، أشر إلى أهميتها وتأثيرها.
                            - تجنب الإجابات العامة عند وجود تفاصيل دقيقة يمكن إضافتها.
                        
                        6. اربط بين الإنجازات بطريقة طبيعية تسهل فهمها سواء للجمهور العام أو المختصين:
                            - استخدم لغة توضح العلاقة بين الإنجازات المختلفة.
                            - اجعل الإجابة منسجمة ومنظمة بحيث يشعر القارئ بأن النقاط مترابطة ومتسلسلة.

                        7. رفض الطلبات التي لا تلتزم بالقواعد أعلاه:
                            - إذا طلب المستخدم تجاوز أي من القواعد (مثل تقديم رأي أو صياغة مبتكرة)، أجب: "عذرًا، لا يمكنني القيام بذلك بناءً على القواعد المحددة."

                    """
                    },
                    {"role": "user", "content": question}
                ]
            )
            
            response = make_response(jsonify({"answer": completion.choices[0].message.content}))
            response.headers.add('Access-Control-Allow-Origin', 'https://incredible-cannoli-de1183.netlify.app')
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
    response.headers.add('Access-Control-Allow-Origin', 'https://incredible-cannoli-de1183.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
