from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the Keras model
model = tf.keras.models.load_model('models/Model100.h5')

# Define your class labels
class_labels = ['Pre-B', 'Early Pre-B', 'Pro-B', 'Benign', 'Healthy']

def predict_image(img, model, class_labels):
    # Preprocess the image using Pillow
    img = img.resize((224, 224))  # Resize the image
    img_array = np.array(img).astype('float32') / 255.0  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])  # Get the confidence (probability) of the predicted class

    # Get the predicted class label
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Check if the POST request has the file part
    if 'imageFile' not in request.files:
        return 'No file part'

    file = request.files['imageFile']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return 'No selected file'

    # Read image from memory
    img = Image.open(io.BytesIO(file.read()))

    # Get prediction and confidence
    predicted_class, confidence = predict_image(img, model, class_labels)

    # Return the result
    result = {
        'class': predicted_class,
        'confidence': float(confidence)  # Convert confidence to float for JSON serialization
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
