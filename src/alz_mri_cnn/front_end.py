from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import keras
import cv2
import numpy as np
import random
import shutil

IMG_SIZE = (128, 128)
# IMG_SIZE = (128//2, 128//2)

RUNNING_DIR = "/tmp/alz_mri_cnn/"

NICER_CLASS_NAMES = [
    "Mild Impairment",
    "No Impairment",
    "Moderate Impairment",
    "Very Mild Impairment",
]
CLASSES = [
    "MildDemented",
    "NonDemented",
    "ModerateDemented",
    "VeryMildDemented",
]


app = Flask(__name__)

model_accuracy = None
best_performing_model = (
    "alz_cnn_98%_acc_25_es_32_bs_0.001_lr_99%_data_0.05_loss_86_seconds.keras"
)
model_accuracy = "98%"

model = None
 
def get_model() -> keras.Model:
    global model
    if not model:
        if os.path.exists(best_performing_model):
            model_name = os.path.join(RUNNING_DIR, 'models', best_performing_model)
        else:
            # grab a model
            models = []
            best = None
            for model in os.listdir(os.path.join(RUNNING_DIR, 'models')):
                if '.keras' in model:
                    models.append(model)
                    acc = int(model[model.find("%")-2:model.find("%")].replace("_",""))
                    if best is None or acc > best[0]: 
                        best = (acc,model)
            
            model_name = os.path.join(RUNNING_DIR, 'models', best[1])

        model = keras.models.load_model(model_name)
    return model


@app.route("/", methods=["POST", "GET"])
def on_start():
    """
    The homepage for this app - just display index.html. On a post (click) of a button (request.form.get('impairment_val')) show a random image
        of that class and the prediction the model gave.
    """
    if request.method == "POST":
        image, prediction, confidence = get_random_of_class(request.form.get("impairment_val"))
    elif request.method == "GET":
        return render_template("index.html")

    # Clear out the static dir of all previous entries (jpgs)
    for p in os.listdir("static"):
        if ".jpg" in p:
            try:
                os.remove(f"static/{p}")
            except Exception as e:
                print(f"encountered issue when removing image {p} from static dir: {e}")
    shutil.copy(image, "static")

    # Render the image now that it is in the static dir
    index = image.rindex("/")
    img_location = os.path.join('static', image[index+1 : len(image)])
    return render_template(
        "index.html",
        model_accuracy=model_accuracy,
        result=prediction,
        image=img_location,
        confidence=confidence
    )


def get_random_of_class(chosen_class):
    """
    With the specified chosen_class, find a random image and get a prediction of that image from the model.
    """
    if chosen_class in NICER_CLASS_NAMES:
        index = NICER_CLASS_NAMES.index(chosen_class)
    elif chosen_class in CLASSES:
        index = CLASSES.index(chosen_class)
    
    dir = os.path.join(RUNNING_DIR, 'data', 'test')
    for path in os.listdir(dir): 
        if path == CLASSES[index]:
            images = os.listdir(os.path.join(dir,path))
    assert images
    image = random.choice(images)
    return predict_image(os.path.join(dir, CLASSES[index], image))


def predict_image(path):
    """
    Given an image at the specified path, feed it to the best-performing model and return the (path of the image,
        class the model predicted, found probability for that predicted class).
    """
    image = cv2.imread(path)
    image_resize = cv2.resize(image, IMG_SIZE)
    data = np.asanyarray(image_resize, dtype=float)
    x_data = np.asarray(data) / (255.0)  # type: np.typing.NDArray[np.float64]
    x_data_reshape = np.reshape(x_data, (1, IMG_SIZE[0], IMG_SIZE[1], 3))
    probabilities = get_model().predict(x_data_reshape)
    max = np.argmax(probabilities)
    return (path, NICER_CLASS_NAMES[max], int(probabilities[0][max]*100))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Given a POST to the /predict endpoint, attempt to find a file in the request. If a file is found, run that file through the predictive
        model and output the resulting predictions.
    """
    if "file" not in request.files:
        return "Unable to process file if no file sent"
    files = request.files.items(multi=True)
    vals = []
    for f in files:
        # read and save file to output directory
        file = f[1]
        filename = secure_filename(file.filename)  # type: ignore
        filepath = os.path.join('output', filename)
        file.save(filepath)
        # predict image, remove file
        vals.append(predict_image(filepath))
        os.remove(filepath)
    return vals

if __name__ == "__main__":
    model = get_model()
    app.run(debug=True)
