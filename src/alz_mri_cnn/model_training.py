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

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(2)

"""
Dataset source:
    https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy
"""


"""
WGANs-GP to increase sizes of small samples for mild impairment cases
"""


def reduce_size_of_dataset(percent_of_data: float, x_train, y_train, x_test, y_test, x_cv, y_cv):
    print("shuffling training data and indices")
    train_indices = np.arange(int(percent_of_data * x_train.shape[0]))
    np.random.shuffle(train_indices)

    print("shuffling testing data and indices")
    test_indices = np.arange(int(percent_of_data * x_test.shape[0]))
    np.random.shuffle(test_indices)

    pre_reduce_samples = x_train.shape[0]
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
    PATH = "data/"
    train = ImageDataset(PATH=f"{PATH}/train", TRAIN=True)
    test = ImageDataset(PATH=f"{PATH}/test", TRAIN=False)

    # Load dataset
    x_train, y_train = train.load_data()
    x_test, y_test = test.load_data()
    x_cv, x_test = np.array_split(x_test, 2)
    y_cv, y_test = np.array_split(y_test, 2)

    x_train, y_train, x_test, y_test, x_cv, y_cv = reduce_size_of_dataset(
        percent_of_data, x_train, y_train, x_test, y_test, x_cv, y_cv
    )

    num_classes = train.get_num_categories()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_cv = tf.keras.utils.to_categorical(y_cv, num_classes)

    return num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv


class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("acc") >= 0.995:
            self.model.stop_training = True
        pass


def create_model(num_classes):
    tf.random.set_seed(1234)

    # Create tensorflow model
    model = Sequential(
        [
            # Convolution and Pooling 3 times before flattening to reduce total number of pixels passed to last Dense layer
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(128, 128, 3)
            ),  # relu - ReLU(x)=max(0,x)
            MaxPooling2D(2, 2),
            # Dropout(0.15),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            # Dropout(0.25),                                                  # select a percentage of outputs at random and set to 0, helps regularize
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

    print(model.summary())

    return model


def train_model(
    percent_of_data=0.99, num_epochs=25, batch_size=32, learning_rate=0.001
):
    model = None
    try:
        start = time.time()
        num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv = load_data(
            percent_of_data
        )
        model = create_model(num_classes)

        # Compile model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["acc"],
        )

        model.build()
        print(f"Model params:{model.count_params()}")

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
        loss, acc = model.evaluate(x_test, y_test, verbose=0)  # type: ignore

        print("Test loss:", loss)
        print("Test accuracy:", acc)
        end = time.time()
        elapsed_time = f"{(end-start):.0f}"
        acc_perc = f"{int(acc*100)}%"
        data_perc = f"{int(percent_of_data*100)}%"

        f = open("logs/histories.txt", "a")
        out = f"{acc_perc}, {data_perc}, {batch_size}, {learning_rate}, {history.params}, {history.history['acc']}, {history.history['loss']}, {elapsed_time}\n"
        if acc >= 0.95:
            model.save(
                f"models/95-99/alz_cnn_{acc_perc}_acc_{num_epochs}_es_{batch_size}_bs_{learning_rate}_lr_{data_perc}_data_{loss:.2f}_loss_{elapsed_time}_seconds.keras"
            )
            f.write(out)
            f.close()
        elif acc >= 0.5:
            f.write(out)
            f.close()
        else:
            print(f"skipping write of model with acc:{acc_perc}")

        del history
    except Exception as ex:
        # write out failure
        f = open("logs/failures.txt", "a")
        f.write(f"{(percent_of_data,batch_size,num_epochs)}, {ex}\n")
        f.close()
        time.sleep(5)
    finally:
        # garbage collection
        if model:
            del model
        K.clear_session()
        gc.collect()
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)

    f = open("logs/histories.txt", "a")
    f.write(f"\ntest run: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
    f.write(
        "accuracy percent,percent of data,batch size,learning_rate,history.params,history.history['acc'],history.history['loss'],elapsed_time\n"
    )

    f.close()

    # plot features
    #   no plotting features needed - dataset consists of images
    #       features are just intensity values of pixels

    data_subets = [0.99]  # may need to downsample images before using more data
    epochs = [10, 25, 50, 75, 100]  # 'random' scaling numbers
    batch_sizes = [32]  # powers of 2
    learning_rates = [0.001, 0.01, 0.1, 1]
    for data_sub in data_subets:
        for epoch in epochs:
            for batch in batch_sizes:
                for learn_rate in learning_rates:
                    train_model(data_sub, epoch, batch, learn_rate)


if __name__ == "__main__":
    main()
