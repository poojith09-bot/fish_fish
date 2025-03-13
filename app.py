from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import os

app = Flask(__name__)

# Ensure the uploadimages directory exists
UPLOAD_FOLDER = "uploadimages"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model
MODEL_PATH = "models/fish_disease_recog_model_pwp.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels
labels = [
    'Streptococcosis', 
    'Parasitic Diseases', 
    'Columnaris Disease', 
    'Tilapia Lake Virus', 
    'Motile Aeromonad Septicemia', 
    'Normal Nile Tilapia'
]

# Load disease details
FISH_DISEASE_JSON = "fish_disease.json"
if not os.path.exists(FISH_DISEASE_JSON):
    raise FileNotFoundError(f"Disease info file not found: {FISH_DISEASE_JSON}")

with open(FISH_DISEASE_JSON, 'r') as file:
    fish_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image_path):
    """ Load and preprocess image """
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)  # Expand for model
    return feature

def model_predict(image_path):
    """ Predict fish disease """
    img = extract_features(image_path)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    # Fetch disease details
    if isinstance(fish_disease, list):
        disease_info = next((item for item in fish_disease if item["name"] == predicted_label), None)
    else:
        disease_info = fish_disease.get(predicted_label, {
            "name": predicted_label,
            "cause": "Information not available",
            "cure": "No specific cure found"
        })

    return disease_info

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        if 'img' not in request.files:
            return "No file uploaded", 400
        
        image = request.files['img']
        if image.filename == '':
            return "No file selected", 400

        # Generate unique filename and save image
        filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(file_path)

        # Predict disease
        prediction = model_predict(file_path)

        # Pass correct image URL to template
        image_url = url_for('uploaded_images', filename=filename)

        return render_template('home.html', result=True, imagepath=image_url, prediction=prediction)
    
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)




 
