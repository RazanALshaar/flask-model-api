from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# إعداد Flask
app = Flask(__name__)

# تحميل النموذج المدرب مرة واحدة
model = load_model("/content/drive/MyDrive/model_xception5.h5")

# دالة لتحضير الصورة
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # حفظ الصورة مؤقتاً
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    # تجهيز الصورة والتنبؤ
    try:
        img_array = prepare_image(temp_path)
        predictions = model.predict(img_array)
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({
        'predicted_class_index': predicted_class_index,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
