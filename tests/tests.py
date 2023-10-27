import logging
import os
import time
import unittest
from threading import Thread

import pytest
import requests_mock

from alz_mri_cnn.front_end import (get_categories, get_model,
                                   get_random_image_of_class, shutdown,
                                   start_local_server)
from alz_mri_cnn.model_training import init, load_data, train_model

LOGGER = logging.getLogger(__name__)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
IN_TOX = os.getenv("TOX_PACKAGE") is not None
RUNNING_DIR = "/tmp/alz_mri_cnn/"
NUM_EPOCHS = 1
URL = "http://127.0.0.1:5000/"


class TestMethods(unittest.TestCase):
    # @pytest.fixture
    # def client():
    #     app.config.update({"TESTING": True})
    #     with app.test_client() as client:
    #         yield client

    """TODO"""

    @pytest.mark.run(order=1)
    def test_init(self):
        os.chdir(RUNNING_DIR)
        assert init()

    """TODO"""

    @pytest.mark.run(order=1)
    def test_load_data(self):
        assert load_data()

    """ Test how the frontend would load its keras model """

    @pytest.mark.run(order=1)
    def test_load_keras_model(self):
        assert get_model()

    """ Attempt to train the model using a default set of parameters """

    @pytest.mark.run(order=2)
    @unittest.skipIf(
        IN_TOX or IN_GITHUB_ACTIONS,
        "not supported in tox/github actions that do not support CUDA",
    )
    def test_train_model(self):
        assert train_model(
            num_epochs=NUM_EPOCHS, percent_of_data=0.5, force_save=True, show_plot=False
        )
        pass

    """TODO"""

    @pytest.mark.run(order=1)
    def test_start_ui(self):
        def send_delayed_shutdown():
            time.sleep(2)
            with requests_mock.Mocker() as rm:
                response = rm.get(f"{URL}shutdown")
                print(f"result:{response}")

        shutdown_thread = Thread(target=send_delayed_shutdown)
        shutdown_thread.start()
        start_local_server()
        shutdown_thread.join()
        shutdown()

    # """ TODO """

    # @pytest.mark.run(order=1)
    # def test_on_start(self):
    #     start_local_server()
    #     on_start()
    #     shutdown()

    """ Attempt to train the model using a default set of parameters """

    @pytest.mark.run(order=3)
    def test_random_of_class(self):
        for c in get_categories():
            assert get_random_image_of_class(c)

    """TODO"""

    @pytest.mark.run(order=3)
    def test_predict_endpoint(self):
        start_local_server()

        file = get_random_image_of_class(get_categories()[0])

        with requests_mock.Mocker() as rm:
            rm.post(f"{URL}predict", body={"file": open(file, "rb")})

        shutdown()
