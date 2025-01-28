from PIL.Image import Image
from flask import Flask,jsonify, request
import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from time import time
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Load the saved model
#with open('hdbscan_model.pkl', 'rb') as model_file:
model_version = os.getenv('MODEL_VERSION', 'v1')
model_path = f'model/hdbscan_model_{model_version}.pkl'
logger.info('Loading the saved model')
model = joblib.load('model/hdbscan_model.pkl')
logger.info('Model loaded successfully')

@app.before_request
def start_timer():
    request.start_time = time()

@app.after_request
def log_response(response):
    duration = time() - request.start_time
    logger.info(f"{request.method} {request.path} {response.status} {duration:.4f}s")
    return response

@app.route('/predict', methods=['POST'])
def predict_cluster():
    logger.info('Received a prediction request')
    try:
        file = request.files['image']
        logger.debug('File received: %s', file.filename)

        # Convert the uploaded file to a NumPy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        logger.debug('File converted to NumPy array')

        # Decode the NumPy array as an OpenCV image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        logger.debug('Image decoded using OpenCV')

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug('Image converted to grayscale')

        image = cv2.resize(grayscale_image,(250, 250))
        logger.debug('Image resized to 250x250')

        image_array = np.array(image).flatten()
        logger.debug('Image flattened')

        # Preprocess with scaler and PCA
        scaler = StandardScaler()
        pca = PCA()
        image_scaled = scaler.transform([image_array])
        logger.debug('Image scaled using StandardScaler')

        image_pca = pca.transform(image_scaled)
        logger.debug('Image transformed using PCA')

        logger.debug('Image processed and ready for prediction')
        # Use your clustering or prediction logic
        cluster_label = model.predict(image_pca)  # Replace `clusterer` with your HDBSCAN object
        logger.info('Prediction made successfully')

        return jsonify({'cluster_label': int(cluster_label[0])})
    
    except Exception as e:
        logger.error('Error during prediction: %s', str(e))
        return 'Error during prediction', 500
    

if __name__ == '__main__':
    logger.info('Starting the Flask app')
    app.run(debug=True, host="0.0.0.0", port=5000)
    logger.info('Flask app is running')