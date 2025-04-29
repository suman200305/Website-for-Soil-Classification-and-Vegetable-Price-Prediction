from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from pymongo import MongoClient
import bcrypt
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Flask setup
app = Flask(__name__)
app.secret_key = 'd4a92bb4544a18d31a7a908284a2a357'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['user_database']
users_collection = db['users']

# Load vegetable price dataset
def load_data():
    data = pd.read_csv('vegetables_prices_additional_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)
    return data

data = load_data()

# Price forecasting logic
def forecast_prices(state, district, market, vegetable, start_date):
    filtered_data = data[
        (data['State'] == state) &
        (data['District'] == district) &
        (data['Market'] == market) &
        (data['Vegetable'] == vegetable)
    ]
    
    if filtered_data.empty:
        return None, None

    filtered_data.set_index('Date', inplace=True)
    target = filtered_data['Modal_Price']

    model = SARIMAX(target, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    result = model.fit(disp=False)

    forecast_dates = pd.date_range(start=pd.to_datetime(start_date), periods=15)
    forecast = result.get_forecast(steps=15)
    forecast_values = forecast.predicted_mean

    return forecast_dates, forecast_values

# Load soil model once
soil_model = tf.keras.models.load_model("soil_classifier_model.h5")
CLASS_NAMES = ['Alluvial soil', 'Clayey soils', 'Laterite soil', 'Loamy soil', 'Sandy loam', 'Sandy soil']

# Predict soil type from image
def predict_soil_type(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    predictions = soil_model.predict(img_array)

    if predictions.ndim == 2 and predictions.shape[0] == 1:
        predictions = predictions[0]  # flatten if batch size = 1

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = round(float(predictions[predicted_index]) * 100, 2)
    return predicted_class, confidence

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    mobile = request.form.get('signupMobile')
    password = request.form.get('signupPassword')
    confirm_password = request.form.get('signupConfirmPassword')

    if password != confirm_password:
        flash("Passwords do not match!")
        return redirect(url_for('home'))

    if users_collection.find_one({'mobile': mobile}):
        flash("Mobile number is already registered!")
        return redirect(url_for('home'))

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({'mobile': mobile, 'password': hashed_password})
    flash("Sign up successful! You can now log in.")
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    mobile = request.form.get('loginMobile')
    password = request.form.get('loginPassword')

    user = users_collection.find_one({'mobile': mobile})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        flash("Login successful!")
        return redirect(url_for('home'))
    else:
        flash("Invalid mobile number or password!")
        return redirect(url_for('home'))

@app.route('/predict', methods=['GET'])
def show_predict_page():
    return render_template('predict.html', forecast_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get('state')
    district = request.form.get('district')
    market = request.form.get('market')
    vegetable = request.form.get('vegetable')
    start_date = request.form.get('start_date')

    forecast_dates, forecast_values = forecast_prices(state, district, market, vegetable, start_date)

    if forecast_dates is None or forecast_values is None:
        return render_template('predict.html', error="No data available for the given inputs.", forecast_data=None)

    forecast_data = [
        {'date': str(date.date()), 'predicted_price': round(price, 2)}
        for date, price in zip(forecast_dates, forecast_values)
    ]

    return render_template('predict.html', forecast_data=forecast_data, error=None)

# Soil Prediction Route
@app.route('/templates/service2.html', methods=['GET', 'POST'])
def service2():
    predicted_soil = None
    soil_info = None
    image_url = None
    confidence = None

    if request.method == 'POST':
        if 'soilImage' not in request.files:
            return render_template('service2.html', prediction="No file uploaded.")

        file = request.files['soilImage']
        if file.filename == '':
            return render_template('service2.html', prediction="No file selected.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_soil,confidence = predict_soil_type(filepath)
        image_url = f'/static/uploads/{filename}'  # for displaying the uploaded image

        characteristics = {
    'Alluvial soil': [
        "1. Rich in nutrients, ideal for agriculture.",
        "2. Found in river valleys and deltas, especially near rivers and lakes.",
        "3. Fertile and easy to work with, supports high crop yields.",
        "4. Well-drained, ensuring optimal plant growth.",
        "5. Commonly used for growing rice, wheat, and other crops.",
        "6. Soil texture varies but is usually a mix of sand, silt, and clay.",
        "7. Rice, Wheat, Sugarcane, Pulses, Jute, Mustard, Potato, Onion, Tomato, Maize, Cotton, Barley"
        " ",
         "1. पोषक तत्वों से भरपूर, कृषि के लिए उपयुक्त।",
            "2. नदी घाटियों और डेल्टाओं में पाई जाती है, विशेष रूप से नदियों और झीलों के पास।",
            "3. उपजाऊ और काम करने में आसान, उच्च फसल उत्पादन का समर्थन करती है।",
            "4. अच्छी जल निकासी, जिससे पौधों की वृद्धि बेहतर होती है।",
            "5. चावल, गेहूं और अन्य फसलों की खेती के लिए आमतौर पर उपयोग की जाती है।",
            "6. मिट्टी की बनावट अलग-अलग हो सकती है लेकिन आमतौर पर रेत, गाद और मिट्टी का मिश्रण होती है।",
            " ",
            "1. পুষ্টিতে সমৃদ্ধ, চাষাবাদের জন্য আদর্শ।",
            "2. নদী উপত্যকা এবং বদ্বীপ অঞ্চলে পাওয়া যায়, বিশেষ করে নদী ও হ্রদের আশেপাশে।",
            "3. উর্বর ও ব্যবহার করা সহজ, উচ্চ ফলনের জন্য সহায়ক।",
            "4. ভালোভাবে নিষ্কাশিত, যা উদ্ভিদের সঠিক বৃদ্ধির জন্য সহায়ক।",
            "5. সাধারণত ধান, গম এবং অন্যান্য ফসলের জন্য ব্যবহৃত হয়।",
            "6. মাটির গঠন সাধারণত বালি, সিল্ট এবং কাদার মিশ্রণ।"
    ],
    
    'Clayey soils': [
        "1. High water retention, which may lead to poor drainage.",
        "2. Sticky when wet and hard when dry, making it difficult to work with.",
        "3. Rich in nutrients but may be prone to waterlogging.",
        "4. Typically found in low-lying areas and riverbanks.",
        "5. Often requires good soil management to avoid compaction.",
        "6. Best suited for crops that thrive in moisture-rich environments.",
        "7. Paddy, Soybean, Ragi, Lentils, Mustard, Gram, Sugar beet, Broccoli, Cauliflower, Brinjal, Turmeric, Arhar",
        " ",
        "1. उच्च जलधारण क्षमता, जिससे जल निकासी में कठिनाई हो सकती है।",
            "2. गीला होने पर चिपचिपा और सूखने पर कठोर, जिससे काम करना कठिन हो जाता है।",
            "3. पोषक तत्वों से भरपूर लेकिन जलजमाव की संभावना रहती है।",
            "4. आमतौर पर निचले इलाकों और नदी किनारों में पाई जाती है।",
            "5. संपीड़न से बचने के लिए अच्छी मिट्टी प्रबंधन की आवश्यकता होती है।",
            "6. उन फसलों के लिए उपयुक्त जो अधिक नमी वाली परिस्थितियों में पनपती हैं।",
        " ",
        "1. উচ্চ জলধারণ ক্ষমতা, যা খারাপ নিষ্কাশনের কারণ হতে পারে।",
            "2. ভেজা অবস্থায় আঠালো ও শুকনো হলে শক্ত, ফলে ব্যবহার কঠিন।",
            "3. পুষ্টিতে সমৃদ্ধ হলেও জলাবদ্ধতা হতে পারে।",
            "4. সাধারণত নিচু জমি এবং নদীতীরে পাওয়া যায়।",
            "5. মাটি কঠিন হয়ে যাওয়া রোধ করতে সঠিক ব্যবস্থাপনা প্রয়োজন।",
            "6. সেসব ফসলের জন্য উপযুক্ত যেগুলো আর্দ্র পরিবেশে ভালো জন্মায়।"
    ],
    
    'Laterite soil': [
        "1. Rich in iron and aluminum, giving it a reddish or yellowish color.",
        "2. Highly acidic, which may limit its suitability for some plants.",
        "3. Found in tropical regions with heavy rainfall and high temperatures.",
        "4. Poor in nutrients, so fertilization is often needed for agricultural use.",
        "5. Soil can become hard and compact, requiring deep plowing to improve aeration.",
        "6. Often used for building materials in tropical regions due to its hardening properties.",
        "7. Tea, Coffee, Cashew, Tapioca, Rubber, Coconut, Pineapple, Pepper, Banana, Jackfruit, Millet, Groundnut",
        " ",
        "1. लौह और एल्यूमीनियम से भरपूर, जिससे इसका रंग लाल या पीला होता है।",
            "2. अत्यधिक अम्लीय, जिससे कुछ पौधों के लिए उपयुक्त नहीं होती।",
            "3. उष्णकटिबंधीय क्षेत्रों में भारी वर्षा और उच्च तापमान वाले क्षेत्रों में पाई जाती है।",
            "4. पोषक तत्वों में कमी, इसलिए कृषि के लिए उर्वरकों की आवश्यकता होती है।",
            "5. यह मिट्टी कठोर और सघन हो सकती है, जिसके लिए गहरी जुताई जरूरी होती है।",
            "6. इसकी कठोरता के कारण उष्णकटिबंधीय क्षेत्रों में निर्माण सामग्री के रूप में भी उपयोग होती है।",
        " ",
        "1. লোহা ও অ্যালুমিনিয়ামে সমৃদ্ধ, যার ফলে এর রঙ লালচে বা হলদেটে হয়।",
            "2. অতিমাত্রায় অ্যাসিডিক, ফলে কিছু উদ্ভিদের জন্য উপযুক্ত নয়।",
            "3. উষ্ণমণ্ডলীয় অঞ্চলে অতিবর্ষণ ও উচ্চ তাপমাত্রার এলাকায় পাওয়া যায়।",
            "4. পুষ্টিতে দরিদ্র, তাই চাষের জন্য প্রায়ই সার প্রয়োজন হয়।",
            "5. মাটি শক্ত ও ঘন হতে পারে, এজন্য গভীর চাষ করা প্রয়োজন।",
            "6. এর শক্ত হওয়ার গুণাবলির কারণে গরম অঞ্চলে নির্মাণ সামগ্রী হিসেবেও ব্যবহৃত হয়।"
    ],
    
    'Loamy soil': [
        "1. Considered the ideal soil type for most plant growth.",
        "2. A perfect balance of sand, silt, and clay, providing good aeration.",
        "3. Retains enough moisture while allowing excess water to drain, preventing root rot.",
        "4. Rich in nutrients, supporting healthy plant development.",
        "5. Easy to work with, allowing for deep root growth.",
        "6. Ideal for growing a wide variety of plants, including vegetables, fruits, and flowers.",
        "7. Sugarcane, Cotton, Sunflower, Mustard, Carrot, Lettuce, Tomato, Apple, Banana, Papaya, Wheat, Barley",
        " ",
        "1. अधिकांश पौधों की वृद्धि के लिए आदर्श मानी जाती है।",
            "2. रेत, गाद और मिट्टी का संतुलित मिश्रण, जिससे वायुसंचार अच्छा होता है।",
            "3. नमी को बनाए रखती है और अतिरिक्त पानी को निकाल देती है, जिससे जड़ों को सड़ने से बचाती है।",
            "4. पोषक तत्वों से भरपूर, स्वस्थ पौध विकास को समर्थन करती है।",
            "5. काम करने में आसान, जिससे जड़ों को गहराई तक बढ़ने में मदद मिलती है।",
            "6. सब्जियों, फलों और फूलों सहित कई प्रकार के पौधों की खेती के लिए उपयुक्त।",
        " ",
        "1. অধিকাংশ গাছের বৃদ্ধির জন্য আদর্শ মাটি হিসেবে বিবেচিত হয়।",
            "2. বালি, সিল্ট এবং কাদার সঠিক ভারসাম্য, যা ভালো বায়ুচলাচল নিশ্চিত করে।",
            "3. পর্যাপ্ত আর্দ্রতা ধরে রাখে এবং অতিরিক্ত জল বেরিয়ে যায়, ফলে গাছের শিকড় পচে না।",
            "4. পুষ্টিতে সমৃদ্ধ, স্বাস্থ্যকর গাছপালা বৃদ্ধিতে সহায়তা করে।",
            "5. ব্যবহার করা সহজ, ফলে শিকড় গভীরে প্রবেশ করতে পারে।",
            "6. সবজি, ফল এবং ফুলসহ বিভিন্ন গাছ চাষের জন্য উপযুক্ত।"
    ],
    
    'Sandy loam': [
        "1. Good drainage, preventing waterlogging of roots.",
        "2. Moderate water retention, suitable for most plants.",
        "3. Easy to work with, often requiring less effort for tilling and planting.",
        "4. Holds enough nutrients to support healthy plant growth.",
        "5. Well-suited for crops like vegetables, flowers, and shrubs.",
        "6. May require organic matter addition to improve fertility over time.",
        "7. Groundnut, Potato, Onion, Carrot, Tomato, Cabbage, Garlic, Okra, Cucumber, Beans, Spinach, Chilli",
        " ",
        "1. अच्छी जल निकासी, जिससे जड़ों में जलभराव नहीं होता।",
            "2. मध्यम जलधारण क्षमता, अधिकांश पौधों के लिए उपयुक्त।",
            "3. काम करने में आसान, खेती और रोपण में कम मेहनत लगती है।",
            "4. पौधों की स्वस्थ वृद्धि के लिए पर्याप्त पोषक तत्व रखती है।",
            "5. सब्जियों, फूलों और झाड़ियों के लिए उपयुक्त।",
            "6. समय के साथ उपजाऊता बढ़ाने के लिए जैविक पदार्थ जोड़ना पड़ सकता है।",
        " ",
         "1. ভালো নিষ্কাশন ক্ষমতা, ফলে শিকড়ে জল জমে না।",
            "2. মাঝারি জলধারণ ক্ষমতা, বেশিরভাগ গাছের জন্য উপযুক্ত।",
            "3. ব্যবহার করা সহজ, চাষ এবং রোপণে কম পরিশ্রম লাগে।",
            "4. সুস্থ গাছপালা বৃদ্ধির জন্য পর্যাপ্ত পুষ্টি ধরে রাখতে পারে।",
            "5. সবজি, ফুল ও গুল্মজাত উদ্ভিদের জন্য উপযুক্ত।",
            "6. উর্বরতা বাড়ানোর জন্য মাঝে মাঝে জৈব উপাদান যোগ করার প্রয়োজন হতে পারে।"
    ],
    
    'Sandy soil': [
        "1. Large particles, resulting in excellent drainage.",
        "2. Low nutrient content, requiring regular fertilization for optimal plant growth.",
        "3. Poor water retention, so frequent watering is needed for plants.",
        "4. Light and easy to work with, ideal for crops like melons and root vegetables.",
        "5. Often prone to erosion due to its loose texture and lack of cohesion.",
        "6. Can be amended with organic matter to improve nutrient and moisture retention.",
        "7. Watermelon, Peanuts, Sweet potatoes, Radish, Cucumber, Onion, Carrot, Mustard, Bajra, Cowpea, Bottle gourd, Melon",
        " ",
         "1. बड़े कण, जिससे उत्कृष्ट जल निकासी होती है।",
            "2. पोषक तत्वों की मात्रा कम होती है, इसलिए नियमित उर्वरक की आवश्यकता होती है।",
            "3. जलधारण क्षमता कम होती है, जिससे बार-बार सिंचाई करनी पड़ती है।",
            "4. हल्की और काम करने में आसान, खरबूजा और कंद फसलों के लिए उपयुक्त।",
            "5. इसकी ढीली बनावट और समरूपता की कमी के कारण कटाव की संभावना रहती है।",
            "6. पोषक तत्व और नमी की क्षमता बढ़ाने के लिए इसमें जैविक पदार्थ मिलाया जा सकता है।",
        " ",
        "1. বড় কণার জন্য খুব ভালো নিষ্কাশন ক্ষমতা রয়েছে।",
            "2. পুষ্টি কম থাকে, তাই গাছের উন্নত বৃদ্ধির জন্য নিয়মিত সার প্রয়োজন।",
            "3. জল ধরে রাখতে পারে না, তাই ঘন ঘন জল দিতে হয়।",
            "4. হালকা ও ব্যবহার করা সহজ, তরমুজ ও মূলজাতীয় ফসলের জন্য আদর্শ।",
            "5. এর ঢিলা গঠন এবং সংবদ্ধতার অভাবে প্রায়ই ক্ষয় হয়।",
            "6. পুষ্টি ও আর্দ্রতা ধরে রাখতে জৈব উপাদান মিশিয়ে উন্নত করা যায়।"
    ]
}

        soil_info = characteristics.get(predicted_soil, "No information available.")

    return render_template('service2.html', prediction=predicted_soil, confidence=confidence, soil_info=soil_info, image_url=image_url)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
