import os
import pickle
import random
import numpy as np
import cv2
from typing import Tuple
from tqdm import tqdm
from PIL import Image
import sys

"""
Refactor this code to use keras.preprocessing.image.ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)


...build...compile...

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)
"""


def path_repair():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return "../../" if "src/alz" in script_dir else ""


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
                        random.shuffle(self.image_data)
                    except Exception as e:
                        print(
                            f"exception encountrerd while reading image: {img}. Error: {e}"
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
            pickle_out = open(
                f'{path_repair()}data/X_Data_{"train" if self.TRAIN else "test"}', "wb"
            )
            pickle.dump(X_Data, pickle_out)
            pickle_out.close()

            # Write the Y Label Data
            pickle_out = open(
                f'{path_repair()}data/Y_Data_{"train" if self.TRAIN else "test"}', "wb"
            )
            pickle.dump(Y_Data, pickle_out)
            pickle_out.close()

            print("Pickled Image Successfully ")
            return X_Data, Y_Data

    def load_data(self):
        try:
            # Read the Data from Pickle Object
            X_Temp = open(
                f'{path_repair()}data/X_Data_{"train" if self.TRAIN else "test"}', "rb"
            )
            X_Data = pickle.load(X_Temp)

            Y_Temp = open(
                f'{path_repair()}data/Y_Data_{"train" if self.TRAIN else "test"}', "rb"
            )
            Y_Data = pickle.load(Y_Temp)

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
