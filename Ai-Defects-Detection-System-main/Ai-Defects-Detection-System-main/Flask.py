import io
import tempfile
from flask import Flask, request
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

# load the trained model from file
model = load_model('model_vgg.h5')


# define a route to handle image classification
@app.route('/checkImg', methods=['POST'])
def process_image():
    # get the image data from the request
    image_data = request.files['image'].read()

    # preprocess the image data
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = image.reshape((1, 224, 224, 3))

    prediction = model.predict(image)

    # make a prediction using the loaded model
    predictionStr = (str("defected product") if prediction[0][0] > prediction[0][1] else str("good product"))
    response = {'classification': predictionStr}

    return response



if __name__ == '__main__':
    app.run(port=5554, debug=True,host='0.0.0.0')
