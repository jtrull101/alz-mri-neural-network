from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import keras
import cv2
import numpy as np
import random
import shutil


CLASSES = [
    "Mild Impairment",
    "No Impairment",
    "Moderate Impairment",
    "Very Mild Impairment",
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

model = None
model_accuracy = None
best_performing_model = (
    "alz_cnn_98%_acc_25_es_32_bs_0.001_lr_99%_data_0.05_loss_86_seconds.keras"
)
model_accuracy = "98%"
model_name = f"models/95-99/{best_performing_model}"


def get_model(model_dir=None):
    global model
    global model_name
    if not model:
        if model_dir:
            model_name = model_dir + model_name
        model = keras.models.load_model(model_name)
    return model


@app.route("/", methods=["POST", "GET"])
@app.route("/index", methods=["POST", "GET"])
def on_start():
    """
    The homepage for this app - just display index.html. On a post (click) of a button (request.form.get('impairement_val')) show a random image
        of that class and the prediction the model gave.
    """
    if request.method == "POST":
        image, prediction, confidence = get_random_of_class(request.form.get("impairment_val"), f"{SCRIPT_DIR}/../../")
    elif request.method == "GET":
        return render_template("index.html")

    # Clear out the static dir of all previous entries (jpgs)
    for p in os.listdir(f"{SCRIPT_DIR}/static"):
        if ".jpg" in p:
            try:
                os.remove(f"{SCRIPT_DIR}/static/{p}")
            except Exception as e:
                print(f"encountered issue when removing image {p} from static dir: {e}")
    shutil.copy(image, f"{SCRIPT_DIR}/static")

    # Render the image now that it is in the static dir
    index = image.rindex("/")
    img_location = f"static/{image[index+1 : len(image)]}"
    return render_template(
        "index.html",
        model_accuracy=model_accuracy,
        result=prediction,
        image=img_location,
        confidence=confidence
    )


def get_random_of_class(chosen_class, dir_override=None):
    """
    With the specified chosen_class, find a random image and get a prediction of that image from the model.
    """
    dir = "data/test"
    if dir_override:
        dir = dir_override + dir

    for path in os.listdir(dir):
        if path == chosen_class:
            images = os.listdir(f"{dir}/{path}")
    assert images
    image = random.choice(images)
    return predict_image(f"{dir}/{chosen_class}/{image}")


def predict_image(path):
    """
    Given an image at the specified path, feed it to the best-performing model and return the (path of the image,
        class the model predicted, found probability for that predicted class).
    """
    image = cv2.imread(path)
    image_resize = cv2.resize(image, (128, 128))
    data = np.asanyarray(image_resize, dtype=float)
    x_data = np.asarray(data) / (255.0)  # type: np.typing.NDArray[np.float64]
    x_data_reshape = np.reshape(x_data, (1, 128, 128, 3))
    probabilities = get_model().predict(x_data_reshape)
    max = np.argmax(probabilities)
    # return (path, CLASSES[max], probabilities[max])
    return (path, CLASSES[max], int(probabilities[0][max]*100))


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
        filepath = f"{SCRIPT_DIR}/output/{filename}"
        file.save(filepath)
        # predict image, remove file
        vals.append(predict_image(filepath))
        os.remove(filepath)
    return vals


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    model = get_model(model_dir=f"{SCRIPT_DIR}/../../")
    app.run(debug=True)
