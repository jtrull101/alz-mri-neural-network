# """
# Deprecated code - still intereseting enough to keep
# Data augmentation introduced by the ImageDataGenerator severly hurt the performance of the model
# """

# from sklearn.model_selection import train_test_split

# def collect_images_and_labels_into_dataframe(
#     directory, num_dataframes, percent_of_data
# ):
#     imgs = []
#     labels = []
#     for sub_dir in os.listdir(directory):
#         # list of all image names in the directory
#         image_list = os.listdir(os.path.join(directory, sub_dir))
#         # get full paths for each imag ein list
#         image_list = list(map(lambda x: os.path.join(sub_dir, x), image_list))  # type: ignore
#         # add all images found to mass image list
#         imgs.extend(image_list)
#         # extend labels directory. Use label of sub directory, printed out # of images times
#         labels.extend([sub_dir] * len(image_list))

#     # Create pandas dataframe
#     df = pd.DataFrame({"Images": imgs, "Labels": labels})
#     df = df.sample(frac=1).reset_index(drop=True)  # To shuffle the data
#     test_size = 1.0 / num_dataframes
#     # return portions of dataframe based on the 'num_dataframes' passed in. helpful to separate test/validation data
#     if test_size == 1.0:
#         return df[: int(percent_of_data * len(imgs))], len(imgs)
#     else:
#         v1, v2 = train_test_split(df, test_size=test_size)
#         return (
#             (v1[: int(percent_of_data * len(imgs))], len(imgs)),
#             (v2[: int(percent_of_data * len(imgs))], len(imgs)),
#         )


# def load_data_from_generator(percent_of_data: float = 0.5, batch_size=20):
#     """
#     Load all data from the expected train/ and test/ directories. Returns the number of classes/categories found in the training set, and all
#         x_train, y_train, x_test, y_test, x_cv and y_cv np arrays.
#     """
#     PATH = os.path.join(RUNNING_DIR, "data")
#     train_path = os.path.join(PATH, "train")
#     train_df, len = collect_images_and_labels_into_dataframe(train_path, 1, percent_of_data)

#     # Create ImageDataGenerator objects
#     train_datagen = ImageDataGenerator(rescale=1.0 / 255)
#     train_generator = train_datagen.flow_from_dataframe(
#         dataframe=train_df,
#         directory=train_path,
#         x_col="Images",
#         y_col="Labels",
#         target_size=IMG_SIZE,
#         batch_size=batch_size,
#         class_mode="categorical",
#         shuffle=False
#     )

#     test_path = os.path.join(PATH, "test")
#     (test_df,_), (val_df,_) = collect_images_and_labels_into_dataframe(test_path, 2, percent_of_data)

#     validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
#     validation_generator = validation_datagen.flow_from_dataframe(
#         dataframe=val_df,
#         directory=test_path,
#         x_col="Images",
#         y_col="Labels",
#         target_size=IMG_SIZE,
#         batch_size=batch_size,
#         class_mode="categorical",
#         shuffle=False
#     )

#     test_datagen = ImageDataGenerator(rescale=1.0 / 255)
#     test_generator = test_datagen.flow_from_dataframe(
#         dataframe=test_df,
#         directory=test_path,
#         x_col="Images",
#         y_col="Labels",
#         target_size=IMG_SIZE,
#         batch_size=batch_size,
#         class_mode="categorical",
#     )

#     return (train_generator, len), (validation_generator, None), (test_generator, None)
