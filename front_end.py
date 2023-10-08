from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

import keras
import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
import sys
import random
import shutil

   
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
app = Flask(__name__)
model_accuracy = None

@app.route('/', methods=['POST','GET'])
@app.route('/index', methods=['POST','GET'])
def on_start():
    if request.method == 'POST':
        image, prediction = test_random_of_class(request.form.get('impairment_val'))
        
    elif request.method == 'GET':
        return render_template('index.html')
    
    # anything to be displayed must be in the static dir, ensure only one picture is in there at a time
    for p in os.listdir("static"):
        if '.jpg' in p:
            try:
                os.remove(f"static/{p}")
            except:
                pass
        
    shutil.copy(image, "static/")
    index = image.rindex("/")
    return render_template('index.html', model_accuracy=model_accuracy, result=prediction, image=f"static/{image[index+1 : len(image)]}")    

def test_random_of_class(chosen_class):
    for path in os.listdir("data/test/"):
        if path == chosen_class:
            images = os.listdir(f"data/test/{path}")
    assert images 
    image = random.choice(images)      
    return predict_image(f"data/test/{chosen_class}/{image}")
     
def predict_image(path):
    image = cv2.imread(path)
    image_resize = cv2.resize(image,(128,128))
    data = np.asanyarray(image_resize, dtype=float)
    x_data = np.asarray(data) / (255.0) # Normalize Data
    x_data_reshape = np.reshape(x_data, (1,128,128,3))
    probabilities = model.predict(x_data_reshape)
    max = np.argmax(probabilities)
    return (path, ['Mild Impairment','No Impairment','Moderate Impairment','Very Mild Impairment'][max])
    
@app.route('/predict', methods=['POST']) #type:ignore
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    files = request.files.items(multi=True)
    vals = {}
    
    for f in files:
        # read and save file to output directory
        file = f[1] 
        filename = secure_filename(file.filename) # type: ignore
        filepath = f"{SCRIPT_DIR}/output/{filename}"
        file.save(filepath)
        
        # predict image, remove file
        vals[file.filename] = predict_image(cv2.imread(filepath)) 
        os.remove(filepath)
 
    return vals
    

 
model = None
if __name__ == '__main__':
    os.chdir(SCRIPT_DIR)
    best_performing_model = 'alz_cnn_98%_acc_25_es_32_bs_0.001_lr_99%_data_0.05_loss_86_seconds.keras'
    model_accuracy = "98%"
    model_name = f"{SCRIPT_DIR}/models/95-99/{best_performing_model}"
    
    if not model:
        model = keras.models.load_model(model_name)
        
    app.run(debug=True, host="0.0.0.0", port=3000)