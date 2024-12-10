from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging
import os
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"], supports_credentials=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('server')

# Create OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Add the report content here
REPORT_CONTENT = """
ملخص التقرير السنوي 
التقرير يعكس التزام البنك الإسلامي للتنمية بدوره الإنمائي من خلال التمويل المستدام، الابتكار، وتعزيز الشراكات العالمية لتحقيق أهداف التنمية المستدامة مع التركيز على التكيف مع التحديات الاقتصادية المتغيرة.
يستعرض التقرير السنوي لمدينة الملك عبدالعزيز للعلوم والتقنية لعام 2022 مجموعة متنوعة من الإنجازات والمبادرات التي حققتها المدينة خلال العام، مع التركيز على الابتكار، التطوير التقني، والتعاون الدولي. من أبرز إنجازات المدينة هذا العام تحديث الاستراتيجية المؤسسية بما يتماشى مع الأولويات الوطنية ورؤية المملكة 2030، بهدف تعزيز الابتكار والتنمية المستدامة. كما أطلقت المدينة مبادرات عديدة لدعم الابتكار وريادة الأعمال التقنية، مثل مشروع "الكراج" الذي يوفر بيئة محفزة للمبتكرين والشركات الناشئة لتطوير أفكارهم إلى منتجات تجارية قابلة للتطبيق، مما يعزز مساهمة القطاع الخاص في الاقتصاد الوطني.

وفي إطار تعزيز البحث والتطوير، نفذت المدينة عدة مشاريع تقنية جديدة، من بينها تطوير تقنية مكافحة البعوض المسبب لحمى الضنك، وإطلاق البرنامج السعودي لأشباه الموصلات لبناء قدرات وطنية في هذا المجال الحيوي. كما عززت المدينة تعاونها الدولي مع كبرى الشركات والمراكز البحثية في الولايات المتحدة، اليابان، وكوريا الجنوبية، بهدف نقل المعرفة وتوطين التكنولوجيا وتعزيز الشراكات الاستراتيجية التي ترفع من مكانة المملكة في مجال العلوم والتقنية.

إضافة إلى ذلك، ركزت المدينة على تحسين البنية التحتية للبحث والتطوير وتطوير الكفاءات الوطنية من خلال برامج تدريبية متقدمة، مما يساهم في خلق بيئة عمل محفزة تدعم الإبداع وتعزز القدرة التنافسية للمملكة في مجالات البحث العلمي والتقني. يعكس التقرير التزام مدينة الملك عبدالعزيز للعلوم والتقنية بدعم الابتكار وتحسين جودة الحياة، والمساهمة في تحقيق أهداف رؤية المملكة 2030 من خلال تطوير حلول تقنية متقدمة وتعزيز التعاون بين القطاعين العام والخاص، بما يحقق اقتصادًا مزدهرًا ومجتمعًا معرفيًا متقدمًا.

"""

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

        # Use new OpenAI client syntax
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""أنت مساعد متخصص في الإجابة على الأسئلة المتعلقة بتقرير البنك الإسلامي للتنمية لعام 2018.
                        استخدم المعلومات التالية للإجابة على الأسئلة:
                        
                        {REPORT_CONTENT}
                        """
                    },
                    {"role": "user", "content": question}
                ]
            )
            
            answer = completion.choices[0].message.content
            logger.info(f"Generated answer: {answer}")
            
            return jsonify({"answer": answer})
            
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
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)