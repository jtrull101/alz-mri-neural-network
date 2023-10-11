import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import logging
import gc
import time
from datetime import datetime
from alz_mri_cnn.load_image_data import ImageDataset
import sys
from keras.preprocessing.image import ImageDataGenerator

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(2)

"""
Dataset source:
    https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy
"""


def path_repair():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return "../../" if "src/alz" in script_dir else ""


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


def load_data(percent_of_data: float):
    """
    Load all data from the expected train/ and test/ directories. Returns the number of classes/categories found in the training set, and all
        x_train, y_train, x_test, y_test, x_cv and y_cv np arrays.
    """
    # Create ImageDataset objects
    PATH = f"{path_repair()}data/"
    
    train = ImageDataset(PATH=f"{PATH}/train", TRAIN=True)
    test = ImageDataset(PATH=f"{PATH}/test", TRAIN=False)

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


class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function to stop training if we have achieved a very high accuracy on the training set to avoid overfitting
        """
        if logs.get("acc") >= 0.995:
            self.model.stop_training = True
        pass


def create_model(num_classes):
    """
    Create a Sequential Convolutional Neural Network model that accepts 128x128 rgb images (represented by input_shape (128,128,3)). Convolution and
        Pooling Layers reduce the size of the dataset eventually passed to the Dense layer at the end. Note all Dropout layers have been commented out,
        I've noticed better behavior without these layers.
    """
    # set static seed here for reproducible results
    tf.random.set_seed(1234)

    # Create tensorflow model
    model = Sequential(
        [
            # Convolution and Pooling 3 times before flattening to reduce total number of pixels passed to last Dense layer
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(128, 128, 3)
            ),  # relu - ReLU(x)=max(0,x)
            MaxPooling2D(2, 2),
            # Dropout(0.15),    # select a percentage of outputs at random and set to 0, helps regularize
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            # Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),  # take 4D array, turn into vector
            Dense(128, activation="relu"),
            # Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    # Print the model summary before returning
    print(model.summary())
    return model


def train_model(
    percent_of_data=0.99, num_epochs=25, batch_size=32, learning_rate=0.001
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
    model = None
    try:
        # Load the data for the model
        num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv = load_data(
            percent_of_data
        )

        # Start a timer to capture time to train the model
        start = time.time()

        # Create the model
        model = create_model(num_classes)

        # Compile the model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["acc"],
        )

        # Build the model
        model.build()

        # Fit the model
        callbacks = callback()
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,  # type: ignore
            validation_data=(x_cv, y_cv),
            callbacks=[callbacks],
        )
        end = time.time()
        # Evaluate the model on the test set
        loss, acc = model.evaluate(x_test, y_test, verbose=0)  # type: ignore
        # Get elapsed time from this training, accuracy on test set, and pretty print of percentage of data
        elapsed_time = f"{(end-start):.0f}"
        acc_perc = f"{int(acc*100)}%"
        data_perc = f"{int(percent_of_data*100)}%"

        # Add a log to the histories.log. This is in csv format in case we want to parse this programmatically later
        out = f"{acc_perc}, {data_perc}, {batch_size}, {learning_rate}, {history.params}, {history.history['acc']}, {history.history['loss']}, {elapsed_time}\n"

        if acc >= 0.5:
            # Save the model only if accuracy is over 95%
            if acc >= 0.95:
                model.save(
                    f"models/95-99/alz_cnn_{acc_perc}_acc_{num_epochs}_es_{batch_size}_bs_{learning_rate}_lr_{data_perc}_data_{loss:.2f}_loss_{elapsed_time}_seconds.keras"
                )
            f = open(f"{path_repair()}logs/histories.log", "a")
            f.write(out)
            f.close()
        # Delete the history object for garbage collection
        del history
    except Exception as ex:
        # Write failure and exception to log
        f = open(f"{path_repair()}logs/failures.log", "a")
        f.write(f"{(percent_of_data,batch_size,num_epochs)}, {ex}\n")
        f.close()
        time.sleep(5)
    finally:
        # Garbage collection, delete model, clear keras backend and gc.collect()
        if model:
            del model
        K.clear_session()
        gc.collect()
    return True


def main():
    """
    Main function - perform model training over many iterations
    """
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    # Create starting log, indicating structure of logs
    f = open(f"{path_repair()}logs/histories.log", "a")
    f.write(f"\ntest run: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
    f.write(
        "accuracy percent,percent of data,batch size,learning_rate,history.params,history.history['acc'],history.history['loss'],elapsed_time\n"
    )
    f.close()

    data_subets = [0.99]  # may need to downsample images before using more data
    epochs = [10, 25, 50, 75, 100]  # 'random' scaling numbers
    batch_sizes = [32]  # powers of 2
    learning_rates = [0.001, 0.01, 0.1, 1]
    for data_sub in data_subets:
        for epoch in epochs:
            for batch in batch_sizes:
                for learn_rate in learning_rates:
                    # Train the model over many iterations of the following:
                    #   percentage of data, number of epochs, batch size, and learning rates
                    train_model(data_sub, epoch, batch, learn_rate)


if __name__ == "__main__":
    main()
