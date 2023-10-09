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
# SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

model = None
model_accuracy = None
best_performing_model = (
    "alz_cnn_98%_acc_25_es_32_bs_0.001_lr_99%_data_0.05_loss_86_seconds.keras"
)
model_accuracy = "98%"
model_name = f"models/95-99/{best_performing_model}"


def get_model():
    global model
    if not model:
        model = keras.models.load_model(model_name)
    return model


@app.route("/", methods=["POST", "GET"])
@app.route("/index", methods=["POST", "GET"])
def on_start():
    if request.method == "POST":
        image, prediction = get_random_of_class(request.form.get("impairment_val"))
    elif request.method == "GET":
        return render_template("index.html")

    # anything to be displayed must be in the static dir, ensure only one picture is in there at a time
    for p in os.listdir(f"{SCRIPT_DIR}/static"):
        if ".jpg" in p:
            try:
                os.remove(f"{SCRIPT_DIR}/static/{p}")
            except Exception as e:
                print(f"encountered issue when removing image {p} from static dir: {e}")

    shutil.copy(image, f"{SCRIPT_DIR}/static")
    index = image.rindex("/")
    img_location = f"static/{image[index+1 : len(image)]}"
    return render_template(
        "index.html",
        model_accuracy=model_accuracy,
        result=prediction,
        image=img_location,
    )


def get_random_of_class(chosen_class):
    dir = "data/test"
    for path in os.listdir(dir):
        if path == chosen_class:
            images = os.listdir(f"{dir}/{path}")
    assert images
    image = random.choice(images)
    return predict_image(f"{dir}/{chosen_class}/{image}")


def predict_image(path):
    image = cv2.imread(path)
    image_resize = cv2.resize(image, (128, 128))
    data = np.asanyarray(image_resize, dtype=float)
    x_data = np.asarray(data) / (255.0)  # type: np.typing.NDArray[np.float64]
    x_data_reshape = np.reshape(x_data, (1, 128, 128, 3))
    probabilities = get_model().predict(x_data_reshape)
    max = np.argmax(probabilities)
    return (path, CLASSES[max])


@app.route("/predict", methods=["POST"])  # type:ignore
def predict():
    if "file" not in request.files:
        return "No file part"
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
    model = get_model()
    app.run(debug=True)
