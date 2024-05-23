import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def is_image_file(filename):
    valid_extensions = ['.png', '.jpg', '.jpeg']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def load_and_preprocess_images(files):
    images = []
    for file in files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = np.stack((img,) * 3, axis=-1)
            images.append(img)
    return np.array(images)

def predict_image_outcome(images, model):
    if len(images) > 20:
        # Skip first 10 and last 10 images
        images_to_predict = images[10:-10]
    else:
        images_to_predict = images

    if len(images_to_predict) == 0:
        return {'decision': None, 'count_0': 0, 'count_1': 0}

    predictions = model.predict(images_to_predict)
    predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

    count_0 = np.sum(predicted_labels == 0)
    count_1 = np.sum(predicted_labels == 1)

    decision = 1 if count_1 > count_0 else 0

    return {'decision': decision, 'count_0': count_0, 'count_1': count_1}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        
        if len(files) == 0 or all(f.filename == '' for f in files):
            flash('No selected files')
            return redirect(request.url)
        
        valid_files = [f for f in files if is_image_file(f.filename)]
        
        if len(valid_files) == 0:
            flash('No valid image files selected')
            return redirect(request.url)
        
        for f in valid_files:
            f.seek(0)

        images = load_and_preprocess_images(valid_files)

        model_path = "C:/Users/USER/Desktop/LAST GP CTSCAN 21-5/VGG16_model.h5"
        model = load_model(model_path)

        outcome = predict_image_outcome(images, model)
        
        decision = 'Surgery Needed' if outcome['decision'] == 1 else 'No Surgery Needed'
        count_0 = outcome['count_0']
        count_1 = outcome['count_1']

        return render_template('result.html', decision=decision, count_0=count_0, count_1=count_1)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
