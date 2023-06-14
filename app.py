from flask import Flask, request, jsonify,render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('imageclassifier.h5')


# Preprocess the image
def preprocess_image(image):
    img = cv2.imread(image.filename)
    resize = tf.image.resize(img, (256,256))
    img =  np.expand_dims(resize/255, 0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

# Define the classification route
@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    
    
    # Get the uploaded file
    image = request.files['image']
    
    image_path = f"{image.filename}"
    image.save(image_path)

    # Preprocess the image
    image = preprocess_image(image)
    
    # Make predictions
    
    prediction = model.predict(image)
        
    if prediction > 0.5: 
        result = 'Predicted class is Sad'
    else:
        result = 'Predicted class is Happy'

    
    
    return render_template('index.html',image_path=image_path,result=result)
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
