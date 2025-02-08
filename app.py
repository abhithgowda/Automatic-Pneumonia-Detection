import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from flask import Flask, request, render_template
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Available models
MODEL_PATHS = {
    "VGG16": "models/VGG16_model.h5",
    "VGG19": "models/vgg19_model.h5",
    "ResNet50": "models/optimized_resnet50v2.keras",
    "DenseNet201": "models/densenet201_94plus.keras"
}

# Dictionary to store loaded models (to avoid reloading)
loaded_models = {}

# Labels for classification
labels = ["Normal", "Pneumonia"]

def get_model(model_name):
    """Loads the model if not already loaded"""
    if model_name not in loaded_models:
        loaded_models[model_name] = load_model(MODEL_PATHS[model_name])
    return loaded_models[model_name]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    image_url = None
    selected_model = "VGG16"  # Default model

    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form['model']  # Get selected model

        if file and selected_model in MODEL_PATHS:
            # Save uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Load selected model
            model = get_model(selected_model)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            result = labels[predicted_class]
            image_url = file_path

    return render_template('index.html', result=result, image_url=image_url, selected_model=selected_model, models=MODEL_PATHS.keys())

if __name__ == '__main__':
    app.run(debug=True)
