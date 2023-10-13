import gc
import logging
import os
import pathlib
import pickle
# import kaggle
import shutil
import time
from datetime import datetime
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
# from image_dataset import ImageDataset
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint)
from keras.layers import (Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(2)


IMG_SIZE = (128, 128)
# IMG_SIZE = (128//2, 128//2)
RUNNING_DIR = "/tmp/alz_mri_cnn/"

# DATASET_NAME = "Alzheimer_s Dataset"
DATASET_NAME = "Combined Dataset"


class ImageDataset(object):
    def __init__(self, PATH="", TRAIN=False):
        self.PATH = PATH
        self.TRAIN = TRAIN
        self.NUM_CATEGORIES = 0
        self.WIDTH, self.HEIGHT = None, None

        (
            self.image_data,
            self.x_data,
            self.y_data,
            self.CATEGORIES,
            self.list_categories,
        ) = ([], [], [], [], [])

    def get_num_categories(self) -> int:
        """
        Return the number of categories found in the directory associated with this image dataset
        """
        if self.NUM_CATEGORIES == 0:
            self.get_categories()
        return self.NUM_CATEGORIES

    def get_width_height_from_imgs_in_path(self, path):
        """
        Get the first image in path and return the size associated
        """
        for img in os.listdir(path):
            new_path = os.path.join(path, img)
            return Image.open(new_path).size
        print(f"ERROR: Unable to find any image in path! path={path}")
        assert False

    def get_image_dimensions(self) -> Tuple[int, int]:
        """
        Get the WIDTH and HEIGHT associated with the images in this image dataset. Note this is using an assumption that
            all images will be of the same size. Could use a tensorflow generator to force a consistent size
        """
        if self.WIDTH and self.HEIGHT:
            return self.WIDTH, self.HEIGHT

        while not self.WIDTH or not self.HEIGHT:
            self.CATEGORIES = self.get_categories()
            for c in self.CATEGORIES:
                self.WIDTH, self.HEIGHT = self.get_width_height_from_imgs_in_path(
                    os.path.join(self.PATH, c)
                )
                if self.WIDTH and self.HEIGHT:
                    break

        return self.WIDTH, self.HEIGHT

    def get_categories(self):
        """
        Get all categories that can be found associated with this image dataset. Note that categories will be subdirectories in this dataset
        """
        for path in os.listdir(self.PATH):
            if path not in self.list_categories:
                self.list_categories.append(path)
        print("Found Categories ", self.list_categories, "\n")
        self.NUM_CATEGORIES = len(self.list_categories)
        return self.list_categories

    def process_image(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for c in tqdm(self.CATEGORIES):  # Iterate over categories
                print(f"processing category:{c}")
                folder_path = os.path.join(self.PATH, c)  # Folder Path
                class_index = self.CATEGORIES.index(
                    c
                )  # this will get index for classification

                for img in tqdm(os.listdir(folder_path)):
                    new_path = os.path.join(folder_path, img)  # image Path
                    self.WIDTH, self.HEIGHT = Image.open(new_path).size
                    try:  # if any image is corrupted
                        image_data_temp = cv2.imread(new_path)  # Read Image as numbers
                        image_temp_resize = cv2.resize(
                            image_data_temp, (self.WIDTH, self.HEIGHT)
                        )

                        self.image_data.append([image_temp_resize, class_index])
                    except Exception as e:
                        print(
                            f"exception encountered while reading image: {img}. Error: {e}"
                        )

            data = np.asanyarray(self.image_data, dtype=object)

            print("setting x_data and y_data for data")
            # Iterate over the Data
            for x in tqdm(data):
                self.x_data.append(x[0])  # Get the X_Data
                self.y_data.append(x[1])  # get the label

            X_Data = np.asarray(self.x_data) / (
                255.0
            )  # type: np.typing.NDArray[np.float64]
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.WIDTH, self.HEIGHT, 3)  # type: ignore
            return X_Data, Y_Data
        except Exception as e:
            print(f"failed to run function process_image. exception: {e}")

    def pickle_image(self):
        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        img = self.process_image()
        if img:
            X_Data, Y_Data = img

            # Write the Entire Data into a Pickle File
            x_out = open(
                f'{RUNNING_DIR}data/X_Data_{"train" if self.TRAIN else "test"}', "wb"
            )
            pickle.dump(X_Data, x_out)
            x_out.close()

            # Write the Y Label Data
            y_out = open(
                f'{RUNNING_DIR}data/Y_Data_{"train" if self.TRAIN else "test"}', "wb"
            )
            pickle.dump(Y_Data, y_out)
            y_out.close()

            print("Pickled Image Successfully ")
            return X_Data, Y_Data

    def load_data(self):
        try:
            # Read the Data from Pickle Object
            x_out = open(
                f'{RUNNING_DIR}data/X_Data_{"train" if self.TRAIN else "test"}', "rb"
            )
            X_Data = pickle.load(x_out)
            x_out.close()

            y_out = open(
                f'{RUNNING_DIR}data/Y_Data_{"train" if self.TRAIN else "test"}', "rb"
            )
            Y_Data = pickle.load(y_out)
            y_out.close()

            print("Reading Dataset from Pickle Object")
            return X_Data, Y_Data

        except Exception as e:
            print(f"Could not find pickle file. exception: {e}")
            print("Loading File and Dataset  ...")

            pickled = self.pickle_image()
            if pickled:
                X_Data, Y_Data = pickled
                return X_Data, Y_Data
            else:
                print("Unable to pickle image successfully")
                assert False


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

    # Load dataset
    x_train, y_train = train.load_data()
    x_test, y_test = test.load_data()

    x_cv, x_test = np.array_split(x_test, 2)
    y_cv, y_test = np.array_split(y_test, 2)

    # Take all datasets and reduce them by the percentagle value passed into this function
    x_train, y_train, x_test, y_test, x_cv, y_cv = reduce_size_of_dataset(
        percent_of_data, x_train, y_train, x_test, y_test, x_cv, y_cv)

    # Set train/test/cv Y data to categorical arrays, we are using categorical crossentropy loss
    num_classes = train.get_num_categories()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_cv = tf.keras.utils.to_categorical(y_cv, num_classes)

    return num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv


def collect_images_and_labels_into_dataframe(
    directory, num_dataframes, percent_of_data
):
    imgs = []
    labels = []
    for sub_dir in os.listdir(directory):
        # list of all image names in the directory
        image_list = os.listdir(os.path.join(directory, sub_dir))
        # get full paths for each imag ein list
        image_list = list(map(lambda x: os.path.join(sub_dir, x), image_list))  # type: ignore
        # add all images found to mass image list
        imgs.extend(image_list)
        # extend labels directory. Use label of sub directory, printed out # of images times
        labels.extend([sub_dir] * len(image_list))

    # Create pandas dataframe
    df = pd.DataFrame({"Images": imgs, "Labels": labels})
    df = df.sample(frac=1).reset_index(drop=True)  # To shuffle the data
    test_size = 1.0 / num_dataframes
    # return portions of dataframe based on the 'num_dataframes' passed in. helpful to separate test/validation data
    if test_size == 1.0:
        return df[: int(percent_of_data * len(imgs))], len(imgs)
    else:
        v1, v2 = train_test_split(df, test_size=test_size)
        return (
            (v1[: int(percent_of_data * len(imgs))], len(imgs)),
            (v2[: int(percent_of_data * len(imgs))], len(imgs))
        )


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
        num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv = load_data(percent_of_data)

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

        callbacks = accuracy_stopper()
        # lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch / 20))
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
        optimal_weights_path = os.path.join(RUNNING_DIR, 'models')
        filepath = os.path.join(optimal_weights_path, 'optimal_weights_{val_acc:.0%}.keras')
        val_acc_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', mode='max', save_best_only=True, verbose=1, initial_value_threshold=0.9)
        callback_list = [callbacks, early_stopping, val_acc_checkpoint]
        # Fit the model
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,  # type: ignore
            validation_data=(x_cv, y_cv),
            callbacks=callback_list
        )

        # Plot loss & accuracy over each epoch using matplotlib and seaborn
        df = pd.DataFrame(history.history).rename_axis('epoch').reset_index().melt(id_vars=['epoch'])
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        for ax, mtr in zip(axes.flat, ['loss', 'acc']):
            ax.set_title(f'{mtr.title()} Plot')
            dfTmp = df[df['variable'].str.contains(mtr)]
            sns.lineplot(data=dfTmp, x='epoch', y='value', hue='variable', ax=ax)
        fig.tight_layout()
        plt.show()

        end = time.time()
        # Evaluate the model on the test set
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        # Get elapsed time from this training, accuracy on test set, and pretty print of percentage of data
        elapsed_time = f"{(end-start):.0f}"
        acc_perc = f"{int(acc*100)}%"
        data_perc = f"{int(percent_of_data*100)}%"

        # Add a log to the histories.log. This is in csv format in case we want to parse this programmatically later
        out = f"{acc_perc}, {data_perc}, {batch_size}, {learning_rate}, {history.params}, {history.history['acc']}, {history.history['loss']}, {elapsed_time}\n"

        if force_save or acc >= 0.5:
            # Save the model only if accuracy is over 98%
            if force_save or acc >= 0.98:
                name = f"alz_cnn_{acc_perc}_acc_{num_epochs}_es_{batch_size}_bs_{learning_rate}_lr_{data_perc}_data_{loss:.2f}_loss_{elapsed_time}_seconds.keras"
                model.save(os.path.join(RUNNING_DIR, "models", name), "a")
            f = open(os.path.join(RUNNING_DIR, "logs", "histories.log"), "a")
            f.write(out)
            f.close()
        # Delete the history object for garbage collection
        del history
    except Exception as ex:
        # Write failure and exception to log
        print("logging failure...")
        print(ex)
        f = open(os.path.join(RUNNING_DIR, "logs", "failures.log"), "a")
        f.write(f"{(percent_of_data,batch_size,num_epochs)}, {ex}\n")
        f.close()
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

    # Separate zip into separate directories in data/
    dataset_dir = os.path.join(RUNNING_DIR, DATASET_NAME)
    if pathlib.Path(dataset_dir).exists():
        for dir in os.listdir(dataset_dir):
            if pathlib.Path(os.path.join(RUNNING_DIR, "data", dir)).exists():
                shutil.rmtree(os.path.join(RUNNING_DIR, "data", dir))

            shutil.move(os.path.join(dataset_dir, dir), os.path.join(RUNNING_DIR, "data"))
        os.rmdir(dataset_dir)

    train_dir = os.path.join(RUNNING_DIR, "data", "train")
    test_dir = os.path.join(RUNNING_DIR, "data", "test")
    if pathlib.Path(train_dir).exists():
        assert len(os.listdir(train_dir)) > 0
        return

    # handle previously unzipped data
    dataset_dir = os.path.join(RUNNING_DIR, 'data', DATASET_NAME)
    if pathlib.Path(dataset_dir).exists():
        # Test/train
        for dir in os.listdir(dataset_dir):
            folder = os.path.join(dataset_dir, dir)
            # Impairment level
            for subdir in os.listdir(folder):
                subdir_path = os.path.join(folder, subdir)
                for filename in os.listdir(subdir_path):
                    source = os.path.join(subdir_path, filename, )
                    smaller_dir = os.path.join(RUNNING_DIR, 'data', dir)
                    desired_dir = os.path.join(smaller_dir, subdir)
                    if not os.path.isdir(smaller_dir):
                        os.mkdir(smaller_dir)
                    if not os.path.isdir(desired_dir):
                        os.mkdir(desired_dir)

                    destination = os.path.join(desired_dir, filename)
                    if os.path.isfile(source):
                        shutil.copy(source, destination)

    assert len(os.listdir(train_dir)) > 0
    assert len(os.listdir(test_dir)) > 0


def init():
    required_paths = [
        RUNNING_DIR,
        os.path.join(RUNNING_DIR, "logs"),
        os.path.join(RUNNING_DIR, "data"),
    ]
    for p in required_paths:
        if not os.path.exists(p):
            os.makedirs(p)
        print(f"The new directory {p} is created!")
    os.chdir(RUNNING_DIR)

    # Download data using the Kaggle API
    download_data_from_kaggle()
    return True


def main():
    """
    Main function - perform model training over many iterations
    """
    init()

    # Create starting log, indicating structure of logs
    f = open(os.path.join(RUNNING_DIR, "logs", "histories.log"), "a")
    f.write(f"\ntest run: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
    f.write(
        "accuracy percent,percent of data,batch size,learning_rate,history.params,history.history['acc'],history.history['loss'],elapsed_time\n"
    )
    f.close()

    data_subets = [1]        # may need to downsample images before using more data
    epochs = [250]  # [25, 50, 75, 100]  # 'random' scaling numbers
    batch_sizes = [20]          # powers of 2
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
