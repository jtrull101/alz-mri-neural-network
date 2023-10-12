import unittest
import os
from src.alz_mri_cnn.model_training import train_model, load_data, init
from src.alz_mri_cnn.front_end import get_random_of_class, CLASSES
import pytest 

SCRIPT_DIR = os.path.dirname(__file__)


class TestMethods(unittest.TestCase):
    MODEL_NAME = None
    
    @pytest.mark.run(order=1)
    def test_init(self):
        assert init()
        
    """ TODO """
    @pytest.mark.run(order=1)
    def test_load_data(self):
        assert load_data()
    
    """ Attempt to train the model using a default set of parameters """
    @pytest.mark.run(order=2)
    def test_train_model(self):
        assert train_model(force_save=True)
    
    """ Attempt to train the model using a default set of parameters """
    @pytest.mark.run(order=3)
    def test_random_of_class(self):
        for c in CLASSES:
            assert get_random_of_class(c)


    
