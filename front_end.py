from flask import Flask, request
from werkzeug.utils import secure_filename
import os

import keras
import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
import sys

   
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
app = Flask(__name__)


@app.route('/predict', methods=['POST']) #type:ignore
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    files = request.files.items(multi=True)
    vals = {}
    d = []
    
    for i,f in enumerate(files):
        file = f[1] 
        filename = secure_filename(file.filename) # type: ignore
        filepath = f"{SCRIPT_DIR}/output/{filename}"
        file.save(filepath)
        
        image = cv2.imread(filepath)         
        image_resize = cv2.resize(image,(128,128))
        data = np.asanyarray(image_resize, dtype=float)
        x_data = np.asarray(data) / (255.0) # Normalize Data
        d.append(x_data)
        x_data_reshape = np.reshape(x_data, (1,128,128,3))
        probabilities = model.predict(x_data_reshape)
        max = np.argmax(probabilities)
        vals[file.filename] = ['Mild Impairment','No Impairement','Moderate Impairement','Very Mild Impairement'][max]
 
    return vals
    

 
model = None
if __name__ == '__main__':
    os.chdir(SCRIPT_DIR)
    best_performing_model = 'alz_cnn_98%_acc_25_es_32_bs_0.001_lr_99%_data_0.05_loss_86_seconds.keras'
    model_name = f"{SCRIPT_DIR}/models/95-99/{best_performing_model}"
    
    if not model:
        model = keras.models.load_model(model_name)
        
    app.run(debug=True)