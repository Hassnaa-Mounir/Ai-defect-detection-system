
from flask import Flask ,render_template ,request , jsonify
import pickle 
import numpy as np


app = Flask (__name__)


@app.route('/', methods =['GET'])

def hello_word():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict():
    imagefile= request.files['image'].read()
    image_path = "./images/" + imagefile.filename
    print(image_path)
    imagefile.save(image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port = 3000, debug=True)