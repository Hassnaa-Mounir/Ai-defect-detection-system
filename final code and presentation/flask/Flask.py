import io
from flask import Flask, request
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# load the trained model from file
model = load_model('model.h5')

# define a route to handle image classification
@app.route('/predict', methods=['POST'])
def predict():
    # get the image data from the request
    image_data = request.files['image'].read()
    
    # preprocess the image data
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = image.reshape((1, 224, 224, 3))
    
    # make a prediction using the loaded model
    prediction = model.predict(image)
    
    pred=(str("defected product") if prediction[0][0] > prediction[0][1] else str("good product"))
    # return the prediction to the client
    return {'classification':pred}

if __name__ == '__main__':
    app.run(port = 3000, debug=True)