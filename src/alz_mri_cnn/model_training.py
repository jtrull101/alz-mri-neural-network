import pathlib
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import os
import logging
import gc
import time
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import kaggle
import shutil
from keras.callbacks import LearningRateScheduler

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(2)


IMG_SIZE = (128, 128)
# IMG_SIZE = (128//2, 128//2)
RUNNING_DIR = "/tmp/alz_mri_cnn/"


def collect_images_and_labels_into_dataframe(directory, num_dataframes, percent_of_data):
    imgs = []
    labels = []
    for sub_dir in os.listdir(directory):
        # list of all image names in the directory
        image_list = os.listdir(os.path.join(directory, sub_dir)) 
        # get full paths for each imag ein list
        image_list = list(map(lambda x: os.path.join(sub_dir, x), image_list))
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
        return df[: int(percent_of_data * len(imgs))]
    else:
        v1, v2 = train_test_split(df, test_size=test_size)
        return (
            v1[: int(percent_of_data * len(imgs))],
            v2[: int(percent_of_data * len(imgs))],
        )


def load_data(percent_of_data:float=0.5, batch_size=20):
    """
    Load all data from the expected train/ and test/ directories. Returns the number of classes/categories found in the training set, and all
        x_train, y_train, x_test, y_test, x_cv and y_cv np arrays.
    """
    PATH = os.path.join(RUNNING_DIR, 'data')
    train_path = os.path.join(PATH, 'train')
    df = collect_images_and_labels_into_dataframe(train_path, 1, percent_of_data)

    # Create ImageDataGenerator objects
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=train_path,
        x_col="Images",
        y_col="Labels",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )

    test_path = os.path.join(PATH, 'test')
    test, val = collect_images_and_labels_into_dataframe(test_path, 2, percent_of_data)

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=val,
        directory=test_path,
        x_col="Images",
        y_col="Labels",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory=test_path,
        x_col="Images",
        y_col="Labels",
        target_size=IMG_SIZE,
        batch_size=20,
        class_mode="categorical",
    )

    return train_generator, validation_generator, test_generator


class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function to stop training if we have achieved a very high accuracy on the training set to avoid overfitting
        """
        if logs.get("acc") >= 0.995:
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
    l = list(IMG_SIZE)
    l.append(3) # rgb channels
    input_shape = tuple(l)

    # Create tensorflow model
    model = Sequential(
        [
            # Convolution and Pooling 4 times before flattening to reduce total number of pixels passed to last Dense layer
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),  # relu - ReLU(x)=max(0,x)
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),  # take 4D array, turn into vector
            Dense(IMG_SIZE[0], "relu"),
            Dense(num_classes, "softmax"),
        ]
    )
    # Print the model summary before returning
    print(model.summary())
    return model


def train_model(
    percent_of_data=0.99, num_epochs=25, batch_size=32, learning_rate=0.001, force_save=False,
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
        train_gen, validation_gen, test_gen = load_data(percent_of_data, batch_size)

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

        callbacks = callback()
        # lr_scheduler = LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/20))
        
        # Fit the model
        history = model.fit(
            train_gen,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,  # type: ignore
            validation_data=(validation_gen),
            callbacks=[callbacks], #lr_scheduler],
        )

        end = time.time()
        # Evaluate the model on the test set
        loss, acc = model.evaluate(test_gen, verbose=0)
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
                model.save(os.path.join(RUNNING_DIR, 'models', name), "a")
            f = open(os.path.join(RUNNING_DIR, 'logs', 'histories.log'), "a")
            f.write(out)
            f.close()
        # Delete the history object for garbage collection
        del history
    except Exception as ex:
        # Write failure and exception to log
        print("logging failure...")
        f = open(os.path.join(RUNNING_DIR, 'logs', 'failures.log'), "a")
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
    # Download dataset from kaggle
    kaggle.api.read_config_environment()
    kaggle.api.authenticate()
    try:
        kaggle.api.dataset_download_files('tourist55/alzheimers-dataset-4-class-of-images', path=RUNNING_DIR, unzip=True)
    except Exception as e:
        print("Unable to download dataset from kaggle, check ~/.kaggle/kaggle.json has active credentials")
        raise e
    
    # Separate zip into separate directories in data/
    dataset_dir = os.path.join(RUNNING_DIR, 'Alzheimer_s Dataset') 
    for dir in os.listdir(dataset_dir): 
        if pathlib.Path(os.path.join(RUNNING_DIR, 'data', dir)).exists():
            shutil.rmtree(os.path.join(RUNNING_DIR, 'data', dir))
        
        shutil.move(os.path.join(dataset_dir, dir), os.path.join(RUNNING_DIR, 'data'))
    os.rmdir(dataset_dir)
    
def init():
    required_paths = [
        RUNNING_DIR,
        os.path.join(RUNNING_DIR, 'logs'),
        os.path.join(RUNNING_DIR, 'data')
    ]
    for p in required_paths:
        if not os.path.exists(p): os.makedirs(p)
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
    f = open(os.path.join(RUNNING_DIR, 'logs', 'histories.log'), "a")
    f.write(f"\ntest run: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
    f.write(
        "accuracy percent,percent of data,batch size,learning_rate,history.params,history.history['acc'],history.history['loss'],elapsed_time\n"
    )
    f.close()

    
    data_subets = [0.99]  # may need to downsample images before using more data
    epochs = [10, 25, 50, 75, 100]  # 'random' scaling numbers
    batch_sizes = [32]  # powers of 2
    learning_rates = [0.001, 0.01]
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
