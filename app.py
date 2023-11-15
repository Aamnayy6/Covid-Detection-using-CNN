from tensorflow.keras.preprocessing import image
import numpy as np
from flask import  Flask,render_template, jsonify, request
import pickle
app = Flask(__name__)
from keras.models import load_model
model = load_model("./model.h5")
@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/predict', methods=['POST'])
def predict():
    path = [x for x in request.form.values()]
    print(path)
    img = image.load_img(path[0], target_size=(256, 256))
    img = image.img_to_array(img)/255
    img = np.array([img])
    predictions = (model.predict(img) > 0.5).astype("int32")
    mapped_class = ["Covid" if prediction == 0 else "Normal" for prediction in predictions]
    return render_template('./index.html', prediction_text = 'Image identified as class: {}'.format(mapped_class))


if __name__ == "__main__":
    app.run(debug=True)