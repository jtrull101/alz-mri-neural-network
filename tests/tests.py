import os
import unittest

import pytest

from src.alz_mri_cnn.front_end import NICER_CLASS_NAMES, get_random_of_class
from src.alz_mri_cnn.model_training import init, load_data, train_model

import subprocess

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
IN_TOX = os.getenv("TOX_PACKAGE") is not None


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMethods, self).__init__(*args, **kwargs)
        subprocess.call(['sh', './init.sh'])

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
        IN_TOX or IN_GITHUB_ACTIONS,
        "not supported in tox/github actions that do not support CUDA",
    )
    def test_train_model(self):
        assert train_model(num_epochs=10, percent_of_data=0.5, force_save=True)

    """ Attempt to train the model using a default set of parameters """

    @pytest.mark.run(order=3)
    def test_random_of_class(self):
        for c in NICER_CLASS_NAMES:
            assert get_random_of_class(c)
