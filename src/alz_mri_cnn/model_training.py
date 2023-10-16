import gc
import logging
import os
import pathlib
import pickle

# import kaggle
import shutil
import time
from datetime import datetime

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# from image_dataset import ImageDataset
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from pyunpack import Archive

from alz_mri_cnn.image_dataset import ImageDataset
from alz_mri_cnn.utils import DATASET_NAME, IMG_SIZE, RUNNING_DIR

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(2)


LOGGER = logging.getLogger(__name__)


def reduce_size_of_dataset(
    percent_of_data: float, x_train, y_train, x_test, y_test, x_cv, y_cv
):
    """
    Return reduced versions of the x_train, y_train, x_test, y_test, x_cv and y_cv nparrays. The amount that each array will be reduced is
        represented by the precent_of_data float. An input of 0.5 will yield 50% of the initial dataset size. Assert the dataset is reduced by
        checking the size of x_train before and after.
    """
    # Shuffle indices used for training data reduction
    train_indices = np.arange(int(percent_of_data * x_train.shape[0]))
    np.random.shuffle(train_indices)

    # Suffle indices used for test and validation data reduction
    test_indices = np.arange(int(percent_of_data * x_test.shape[0]))
    np.random.shuffle(test_indices)

    pre_reduce_samples = x_train.shape[0]
    # Reduce sizes of datasets then return datasets
    x_train, y_train = x_train[train_indices], y_train[train_indices]
    x_test, y_test, x_cv, y_cv = (
        x_test[test_indices],
        y_test[test_indices],
        x_cv[test_indices],
        y_cv[test_indices],
    )
    assert pre_reduce_samples >= x_train.shape[0]
    return x_train, y_train, x_test, y_test, x_cv, y_cv


def load_data(percent_of_data: float = 0.5):
    # Create ImageDataset objects
    PATH = f"{RUNNING_DIR}/data/"

    train = ImageDataset(PATH=f"{PATH}/Combined Dataset/train", TRAIN=True)
    test = ImageDataset(PATH=f"{PATH}/Combined Dataset/test", TRAIN=False)
    categories = train.get_categories()
    test_categories = test.get_categories()
    assert categories == test_categories
    # write out category file
    with open(os.path.join(PATH, "categories"), "wb") as f:
        pickle.dump(categories, f)

    # Load dataset
    x_train, y_train = train.load_data()
    x_test, y_test = test.load_data()

    x_cv, x_test = np.array_split(x_test, 2)
    y_cv, y_test = np.array_split(y_test, 2)

    # Take all datasets and reduce them by the percentagle value passed into this function
    x_train, y_train, x_test, y_test, x_cv, y_cv = reduce_size_of_dataset(
        percent_of_data, x_train, y_train, x_test, y_test, x_cv, y_cv
    )

    # Set train/test/cv Y data to categorical arrays, we are using categorical crossentropy loss
    num_classes = train.get_num_categories()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_cv = tf.keras.utils.to_categorical(y_cv, num_classes)

    return num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv



class accuracy_stopper(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function to stop training if we have achieved a very high accuracy on the training set to avoid overfitting
        """
        if logs.get("acc") >= 0.995 or logs.get("val_acc") >= 0.995:
            self.model.stop_training = True
        pass


def create_model():
    """
    Create a Sequential Convolutional Neural Network model that accepts 128x128 rgb images (represented by input_shape (128,128,3)). Convolution and
        Pooling Layers reduce the size of the dataset eventually passed to the Dense layer at the end. Note all Dropout layers have been commented out,
        I've noticed better behavior without these layers.
    """
    # set static seed here for reproducible results
    tf.random.set_seed(1234)

    num_classes = 4
    _list = list(IMG_SIZE)
    _list.append(3)  # rgb channels
    input_shape = tuple(_list)

    # Create tensorflow model
    model = Sequential(
        [
            # Convolution and Pooling 4 times before flattening to reduce total number of pixels passed to last Dense layer
            Conv2D(64, (5, 5), activation="relu", input_shape=input_shape),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(),
            Conv2D(128, (3, 3), activation="relu"),
            Dropout(0.3),
            MaxPooling2D(),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(),
            Flatten(),  # take 4D array, turn into vector
            Dense(IMG_SIZE[0], "relu"),
            Dense(num_classes, "softmax"),
        ]
    )

    # Print the model summary before returning
    print(model.summary())
    return model


def train_model(
    percent_of_data=0.99,
    num_epochs=25,
    batch_size=32,
    learning_rate=0.001,
    force_save=False,
    show_plot=False,
):
    """
    Given the specified percent_of_data, num_epochs, batch_size and learning_rate, first create a model, then compile, build, and finally
        fit the model. The model will be fit and testing using the specified percentage of the whole data subset. Each model is compiled the
        same way, with categorical_crossentropy loss and an Adam optimizer with a configurable learning_rate. After building, the model is fit
        against the training data while validated against the cross validation set. After each epoch the hyper parameters of the model are updated
        in tensorflow in correlation with the result of the loss function assesed against the cross validation set. A callback is included to prevent
        the model from overfitting, stopping the training if our accuracy against the training set exceeds 99.5%.
    At the end of num_epochs, the model is assessed against the percentage subset of the test set. The final loss and accuracy that result from this predict()
        call are used to assess the model in it's trained state. Models with higher than 95% accuracy are saved for consideration as a future best-performing
        model.
    Failures are logged to logs/failures.log and succesful runs are logged in logs/histories.log.
    """
    try:
        # (train_gen, num_train_samples), (validation_gen,_), (test_gen,_) = load_data(percent_of_data, batch_size)
        num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv = load_data(
            percent_of_data
        )

        # Start a timer to capture time to train the model
        start = time.time()

        # Create the model
        model = create_model()

        # Compile the model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["acc"],
        )
        model.build()

        # create callbacks
        acc_stop_callback = accuracy_stopper()
        # lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch / 20))
        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", patience=20, verbose=1
        )
        optimal_weights_path = os.path.join(RUNNING_DIR, "models")
        filepath = os.path.join(
            optimal_weights_path, "optimal_weights_{val_acc:.0%}.keras"
        )
        val_acc_checkpoint = ModelCheckpoint(
            filepath,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=0.9,
        )
        callback_list = [acc_stop_callback, early_stopping, val_acc_checkpoint]

        # Fit the model
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,  # type: ignore
            validation_data=(x_cv, y_cv),
            callbacks=callback_list,
        )

        end = time.time()

        if show_plot:
            # Plot loss & accuracy over each epoch using matplotlib and seaborn
            df = (
                pd.DataFrame(history.history)
                .rename_axis("epoch")
                .reset_index()
                .melt(id_vars=["epoch"])
            )
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            for ax, mtr in zip(axes.flat, ["loss", "acc"]):
                ax.set_title(f"{mtr.title()} Plot")
                dfTmp = df[df["variable"].str.contains(mtr)]
                sns.lineplot(data=dfTmp, x="epoch", y="value", hue="variable", ax=ax)
            fig.tight_layout()
            plt.show()

        # Evaluate the model on the test set
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        # Get elapsed time from this training, accuracy on test set, and pretty print of percentage of data
        elapsed_time = f"{(end-start):.0f}"
        acc_perc = f"{int(acc*100)}%"
        data_perc = f"{int(percent_of_data*100)}%"

        # Add a log to the histories.log. This is in csv format in case we want to parse this programmatically later
        out = f"{acc_perc}, {data_perc}, {batch_size}, {learning_rate}, {history.params}, {history.history['acc']}, {history.history['loss']}, {elapsed_time}\n"

        if force_save or acc >= 0.5:
            with open(os.path.join(RUNNING_DIR, "logs", "histories.log"), "a") as f:
                f.write(out)
        # Delete the history object for garbage collection
        del history
    except Exception as ex:
        # Write failure and exception to log
        print("logging failure...")
        print(ex)
        with open(os.path.join(RUNNING_DIR, "logs", "failures.log"), "a") as f:
            f.write(f"{(percent_of_data,batch_size,num_epochs)}, {ex}\n")
        time.sleep(2.0)
        return False
    finally:
        # Garbage collection, delete model, clear keras backend and gc.collect()
        if model:
            del model
        K.clear_session()
        gc.collect()
    return True


def download_data_from_kaggle():
    # # Download dataset from kaggle
    # kaggle.api.authenticate()
    # try:
    #     kaggle.api.dataset_download_files(
    #         "tourist55/alzheimers-dataset-4-class-of-images",
    #         path=RUNNING_DIR,
    #         unzip=True,
    #     )
    # except Exception as e:
    #     print(
    #         "Unable to download dataset from kaggle, check ~/.kaggle/kaggle.json has active credentials"
    #     )
    #     raise e
    pass


def download_from_google_drive(id, destination):
    url = f"https://docs.google.com/uc?id={id}"
    gdown.download(url, destination, quiet=False)
    print()


def unzip_data():
    desired_dir = os.path.join(RUNNING_DIR, "data")
    zip_dataset_dir = os.path.join(desired_dir, f"{DATASET_NAME}.zip")

    # Separate zip into separate directories in data/
    if pathlib.Path(zip_dataset_dir).exists():
        if not pathlib.Path(desired_dir).exists():
            os.makedirs(desired_dir)

        Archive(zip_dataset_dir).extractall(desired_dir)


def parsed_unzipped_data():
    # handle previously unzipped data
    dataset_dir = os.path.join(RUNNING_DIR, "data", DATASET_NAME)
    if pathlib.Path(dataset_dir).exists():
        # Test/train
        for dir in os.listdir(dataset_dir):
            folder = os.path.join(dataset_dir, dir)
            # Impairment level
            for subdir in os.listdir(folder):
                subdir_path = os.path.join(folder, subdir)
                for filename in os.listdir(subdir_path):
                    source = os.path.join(
                        subdir_path,
                        filename,
                    )
                    smaller_dir = os.path.join(RUNNING_DIR, "data", dir)
                    desired_dir = os.path.join(smaller_dir, subdir)
                    if not os.path.isdir(smaller_dir):
                        os.mkdir(smaller_dir)
                    if not os.path.isdir(desired_dir):
                        os.mkdir(desired_dir)

                    destination = os.path.join(desired_dir, filename)
                    if os.path.isfile(source):
                        shutil.copy(source, destination)


TRAIN_DIR = os.path.join(RUNNING_DIR, "data", "train")
TEST_DIR = os.path.join(RUNNING_DIR, "data", "test")


def init():
    required_paths = [
        RUNNING_DIR,
        os.path.join(RUNNING_DIR, "logs"),
        os.path.join(RUNNING_DIR, "data"),
        os.path.join(RUNNING_DIR, "models"),
    ]
    for p in required_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    required_files = {
        os.path.join(
            required_paths[3], "optimal_weights_98%.keras"
        ): "1U9uywbNatIFAj6XlahT6BBrMqyLgd4qZ",
        os.path.join(
            required_paths[2], "Combined Dataset.zip"
        ): "1SQuB_8IL3s7vZPMeGkOZo116QSTMa6BN",
    }

    # Download data using the Kaggle API
    #   download_data_from_kaggle()

    for k, v in required_files.items():
        download_from_google_drive(v, k)

    # Unzip downloaded data
    unzip_data()

    def check_train_test_dirs():
        if pathlib.Path(TRAIN_DIR).exists() and pathlib.Path(TEST_DIR).exists():
            assert len(os.listdir(TRAIN_DIR)) > 0
            assert len(os.listdir(TEST_DIR)) > 0
            return True
        return False

    if check_train_test_dirs():
        return True

    parsed_unzipped_data()

    if check_train_test_dirs():
        return True

    return False


def main():
    os.chdir(RUNNING_DIR)
    """
    Main function - perform model training over many iterations
    """
    init()

    # Create starting log, indicating structure of log
    with open(os.path.join(RUNNING_DIR, "logs", "histories.log"), "a") as f:
        f.write(f"\ntest run: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
        f.write(
            "accuracy percent,percent of data,batch size,learning_rate,history.params,history.history['acc'],history.history['loss'],elapsed_time\n"
        )

    data_subets = [1]  # may need to downsample images before using more data
    epochs = [250]  # [25, 50, 75, 100]  # 'random' scaling numbers
    batch_sizes = [20]  # powers of 2
    learning_rates = [0.001]
    for data_sub in data_subets:
        for epoch in epochs:
            for batch in batch_sizes:
                for learn_rate in learning_rates:
                    # Train the model over many iterations of the following:
                    #   percentage of data, number of epochs, batch size, and learning rates
                    result = train_model(data_sub, epoch, batch, learn_rate)
    return result


if __name__ == "__main__":
    main()
