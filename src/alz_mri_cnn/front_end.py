import os
import pathlib
import pickle
import random
import shutil
import signal
import typing
from multiprocessing import Process

import cv2
import keras
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from alz_mri_cnn.utils import IMG_SIZE, LOGGER, RUNNING_DIR

CATEGORIES = None

app = Flask(__name__)

model = None
model_accuracy = None


@app.route("/shutdown", methods=["GET"])
def shutdown():
    global server
    LOGGER.debug(f" ==   run server terminate: {server}")
    os.kill(server.pid, signal.SIGKILL)
    LOGGER.debug(f" ==   server terminated: {server}")
    return "Server shutting down..."


def start_local_server():
    global server
    server = Process(target=app.run)
    server.start()
    LOGGER.debug(f" ==   server set: {server}")
    get_model()
    return server


def get_model() -> keras.Model:
    global model
    global model_accuracy
    if not model:
        found_models = []  # type: typing.List[keras.Model]

        def find_model_in_dir(dir):
            for k in os.listdir(dir):
                if ".keras" in k:
                    found_models.append(os.path.join(dir, k))

        best_path_1 = os.path.join(RUNNING_DIR, "models")
        find_model_in_dir(best_path_1)

        best = None
        for model in found_models:
            acc = int(model[model.find("%") - 2 : model.find("%")].replace("_", ""))
            if best is None or acc > best[0]:  # type: ignore
                best = (acc, model)

        model_accuracy, model_name = best  # type: ignore

        LOGGER.debug(
            f"    loading model: {model_name} with accuracy: {model_accuracy}%"
        )
        model = keras.models.load_model(model_name)
    return model


def get_categories(categories=None):
    global CATEGORIES
    if categories and not CATEGORIES:
        CATEGORIES = categories

    if not CATEGORIES:
        category_file = os.path.join(RUNNING_DIR, "data", "categories")
        if pathlib.Path(category_file).exists():
            file = open(category_file, "rb")
            CATEGORIES = pickle.load(file)

        assert CATEGORIES
    return CATEGORIES


def predict_on_request(request):
    class_name = request.form.get("impairment_val")
    LOGGER.debug(f"    searching for random image of class_name: {class_name}")
    image = get_random_image_of_class(class_name)
    prediction, confidence, all_confidence = predict_image(image)
    return image, prediction, confidence, all_confidence


@app.route("/", methods=["POST", "GET"])
def on_start():
    """
    The homepage for this app - just display index.html. On a post (click) of a button (request.form.get('impairment_val')) show a random image
        of that class and the prediction the model gave.
    """
    if not model:
        get_model()

    if request.method == "POST":
        image, prediction, confidence, all_confidence = predict_on_request(request)
    elif request.method == "GET":
        return render_template("index.html")

    # Clear out the static dir of all previous entries (jpgs)
    if image:
        for p in os.listdir("static"):
            if ".jpg" in p:
                try:
                    os.remove(f"static/{p}")
                except Exception as e:
                    LOGGER.debug(
                        f"encountered issue when removing image {p} from static dir: {e}"
                    )
        shutil.copy(image, "static")

        # Render the image now that it is in the static dir
        index = image.rindex("/")
        img_location = os.path.join("static", image[index + 1 : len(image)])
        return render_template(
            "index.html",
            model_accuracy=model_accuracy,
            result=prediction,
            image=img_location,
            confidence=confidence,
            all_confidence=all_confidence,
        )
    return render_template("index.html")


def get_random_image_of_class(chosen_class):
    """
    With the specified chosen_class, find a random image and get a prediction of that image from the model.
    """
    if chosen_class in get_categories():
        index = get_categories().index(chosen_class)

    dir = os.path.join(RUNNING_DIR, "data", "test")
    for path in os.listdir(dir):
        if path == get_categories()[index]:
            images = os.listdir(os.path.join(dir, path))
    assert images
    image = random.choice(images)
    return os.path.join(dir, get_categories()[index], image)


def predict_image(path):
    """
    Given an image at the specified path, feed it to the best-performing model and return the (path of the image,
        class the model predicted, found probability for that predicted class).
    """
    LOGGER.debug(f"predict image at path:{path}")
    image = cv2.imread(path)
    image_resize = cv2.resize(image, IMG_SIZE)
    data = np.asanyarray(image_resize, dtype=float)
    x_data = np.asarray(data) / (255.0)  # type: np.typing.NDArray[np.float64]
    x_data_reshape = np.reshape(x_data, (-1, IMG_SIZE[0], IMG_SIZE[1], 3))
    probabilities = get_model().predict(x_data_reshape)
    max = np.argmax(probabilities)
    probs = {}
    for i, x in enumerate(np.nditer(probabilities)):
        if i != max:
            probs[CATEGORIES[i]] = f"{int(x * 100)}%"  # type:  ignore
        else:
            thisProb = f"{int(x * 100)}%"
    return (get_categories()[max], thisProb, probs)


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
        filepath = os.path.join("output", filename)
        file.save(filepath)
        # predict image, remove file
        vals.append(predict_image(filepath))
        os.remove(filepath)
    return vals


if __name__ == "__main__":
    get_categories()

    global server
    server = start_local_server()

    # frontend must be run from src/alz_mri_cnn dir
    new_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(new_dir)
    LOGGER.debug(f"cd into dir: {new_dir}")
