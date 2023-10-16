import os

IMG_SIZE = (128, 128)
# IMG_SIZE = (128//2, 128//2)

RUNNING_DIR = "/tmp/alz_mri_cnn/"

LOGS_DIR = os.path.join(RUNNING_DIR, "logs")

DATA_DIR = os.path.join(RUNNING_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(RUNNING_DIR, "models")


# DATASET_NAME = "Alzheimer_s Dataset"
DATASET_NAME = "Combined Dataset"

REQUIRED_PATHS = [RUNNING_DIR, LOGS_DIR, DATA_DIR, MODELS_DIR]
