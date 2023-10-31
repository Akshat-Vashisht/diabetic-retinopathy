from flask import Blueprint, render_template, request
import numpy as np
import cv2
from tensorflow import keras
from keras.preprocessing import image
import os



views = Blueprint("views", __name__)
model = keras.models.load_model('../myCNN.model')
predicted_dict = {
    2: "No",
    0: "Mild",
    1: "Moderate",
    3: "Severe",
    4: "Proliferate"
}


def predict_class(img):
    img = cv2.imread(img)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg, (224,224))
    image = np.array(RGBImg) / 255.0
    predict=model.predict(np.array([image]))
    per = np.argmax(predict, axis = 1)
    ser = predicted_dict.get(per[0])
    percentage = predict[0][per][0]*100
    return (ser, percentage)


@views.route("/")
def home():
    return render_template("index.html")

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
def read_image(filename):
    img = keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    return x

@views.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            print(filename)
            file_path = os.path.join('D:/Akshat/Coding/Diabetic Retinopathy/website/Website/static/images/', filename)
            file.save(file_path)
            ser, percentage = predict_class(file_path)
            return render_template('prediction.html', ser = ser,percentage=percentage, filename = filename)
        else:
            return "Unable to read the file. Please check file extension"