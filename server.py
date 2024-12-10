from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import openai
import logging

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"], supports_credentials=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('server')

# Create OpenAI client
client = openai.OpenAI(api_key="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")

# Add the report content here

REPORT_CONTENT = """
تقرير مدينة الملك عبدالعزيز للعلوم والتقنية لعام 2023 يقدم رؤية شاملة حول الإنجازات والتوجهات الاستراتيجية للمدينة، حيث يركز على تعزيز الابتكار والبحث والتطوير والاستدامة بما يتماشى مع رؤية المملكة 2030. يسعى التقرير إلى إبراز دور المدينة كمختبر وطني وواحة للابتكار من خلال التركيز على أولويات البحث والتطوير مثل صحة الإنسان، استدامة البيئة، الريادة في مجالات الطاقة والصناعة، وتعزيز اقتصاديات المستقبل. من أبرز إنجازات المدينة تطوير تقنيات مبتكرة مثل مكافحة البعوض الناقل لحمى الضنك، وتحلية المياه بالطاقة الشمسية، والتقنيات الصحية. كما تم إطلاق مبادرة "الكراج" لدعم التقنيات العميقة، حيث ساعدت في دعم 240 شركة ناشئة من 50 دولة. التقرير يشير أيضًا إلى تنفيذ 35 مشروعًا بحثيًا في مجالات متعددة.

بالإضافة إلى ذلك، تسعى المدينة لتعزيز ريادة الأعمال من خلال دعم الشركات الناشئة والمبتكرين وإنشاء واحات للابتكار. على الصعيد الدولي، عقدت المدينة شراكات مع جهات بحثية وشركات عالمية في الولايات المتحدة واليابان وكوريا الجنوبية، بهدف دعم الابتكار ونقل التكنولوجيا. كما يسلط التقرير الضوء على تجهيز البنية التحتية للمدينة، بما يشمل تطوير المختبرات والمعامل وتدريب الكفاءات الوطنية عبر برامج أكاديمية متقدمة مثل "أكاديمية 32".

حققت المدينة أيضًا جوائز متعددة، مثل جائزة ندلب للتميز وجائزة أفضل مشروع خارج الموقع، إلى جانب تحقيق مراكز متقدمة في المنافسات الدولية. التقرير يُبرز أرقامًا بارزة تشمل 664 خدمة واستشارة تقنية، 307 ورقة علمية منشورة، و3 تقنيات مبتكرة تم توطينها، إضافة إلى 7 شركات منبثقة عن مشاريع البحث والتطوير. يعكس التقرير التزام المدينة بدعم الابتكار وتعزيز الاقتصاد المعرفي، مع مواجهة التحديات وتحقيق الاستدامة الاقتصادية والبيئية في إطار تحقيق أهداف رؤية المملكة 2030.
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
                        "content": f"""أنت مساعد متخصص في تحليل التقارير
                        
                        {REPORT_CONTENT} 

                        قواعد مهمة:
                      1.	اعتمد فقط على المعلومات الموجودة في النص أعلاه:
o	لا تستند إلى أي معلومات خارج النص، حتى لو كانت معروفة أو متوقعة.
o	إذا كان المستخدم يسأل عن موضوع غير موجود في النص، فأجب بوضوح بأن المعلومة غير متوفرة.
2.	أجب باللغة العربية الفصحى:
o	استخدم لغة واضحة ودقيقة خالية من العامية أو الأخطاء النحوية.
o	التزم باستخدام نفس مستوى اللغة الموجود في النص الأصلي.
3.	لا تقدم أي إعادة كتابة أو إعادة صياغة إبداعية أو مختلفة أو مبتكرة للمعلومات:
o	إذا طلب المستخدم إعادة الصياغة أو كتابة الإجابة بطريقة مبتكرة أو بأسلوب مختلف، ارفض الطلب بوضوح.
o	أجب بالنص الأصلي كما هو دون أي تعديل في الأسلوب.
4.	إذا لم تجد المعلومة في النص، قل ذلك بوضوح دون إضافة أو تعديل:
o	لا تضف أي افتراضات أو معلومات إضافية عند الإجابة.
o	الرد يجب أن يكون مباشرًا وواضحًا، مثل: "عذرًا، النص لا يحتوي على هذه المعلومة."
5.	رفض الطلبات التي لا تلتزم بالقواعد أعلاه:
o	إذا طلب المستخدم تجاوز أي من القواعد (مثل تقديم رأي أو صياغة مبتكرة)، أجب: "عذرًا، لا يمكنني القيام بذلك بناءً على القواعد المحددة."


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