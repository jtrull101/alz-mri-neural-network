import unittest
from src.alz_mri_cnn.model_training import train_model, load_data, init
from src.alz_mri_cnn.front_end import NICER_CLASS_NAMES, get_random_of_class
import pytest
import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestMethods(unittest.TestCase):

    """TODO"""

    @pytest.mark.run(order=1)
    def test_init(self):
        assert init()

    """ TODO """

    @pytest.mark.run(order=1)
    def test_load_data(self):
        assert load_data()

    """ Attempt to train the model using a default set of parameters """

    @pytest.mark.run(order=2)
    @unittest.skipIf(
        os.getenv("TOX_PACKAGE") is not None or os.getenv("GITHUB_URL") is not None,
        "not supported in tox/github actions that do not support CUDA",
    )
    def test_train_model(self):
        assert train_model(force_save=True)

    """ Attempt to train the model using a default set of parameters """

    @pytest.mark.run(order=3)
    def test_random_of_class(self):
        for c in NICER_CLASS_NAMES:
            assert get_random_of_class(c)
