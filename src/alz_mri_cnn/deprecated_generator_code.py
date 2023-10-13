

# """
# Deprecated code - still intereseting enough to keep
# Data augmentation introduced by the ImageDataGenerator severly hurt the performance of the model
# """

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
